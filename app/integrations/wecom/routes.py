import logging
import time

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import PlainTextResponse

from app.api.routes import build_rag_service
from app.core.config import get_settings
from app.integrations.wecom.client import WeComClient
from app.integrations.wecom.config import wecom_settings_from_app_settings
from app.integrations.wecom.crypto import (
    WeComCryptoError,
    WeComMessageCrypto,
    build_text_reply_xml,
    parse_xml_message,
    verify_signature,
)
from app.integrations.wecom.handlers import WeComMessageHandler
from app.integrations.wecom.schemas import WeComIncomingMessage
from app.integrations.wecom.user_mapping import JsonWeComUserMappingStore


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/integrations/wecom/callback", response_class=PlainTextResponse)
def verify_wecom_callback(
    msg_signature: str = Query(...),
    timestamp: str = Query(...),
    nonce: str = Query(...),
    echostr: str = Query(...),
):
    settings = wecom_settings_from_app_settings(get_settings())
    _ensure_wecom_enabled(settings.enabled)
    try:
        if settings.encrypted_callbacks_enabled:
            crypto = WeComMessageCrypto(
                token=settings.token,
                encoding_aes_key=settings.encoding_aes_key,
                corp_id=settings.corp_id,
            )
            return crypto.verify_url(
                msg_signature=msg_signature,
                timestamp=timestamp,
                nonce=nonce,
                echostr=echostr,
            )
        if not verify_signature(settings.token, msg_signature, timestamp, nonce, echostr):
            raise WeComCryptoError("Invalid WeCom callback signature.")
        return echostr
    except WeComCryptoError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


@router.post("/integrations/wecom/callback", response_class=PlainTextResponse)
async def receive_wecom_callback(
    request: Request,
    msg_signature: str = Query(...),
    timestamp: str = Query(...),
    nonce: str = Query(...),
):
    settings = wecom_settings_from_app_settings(get_settings())
    _ensure_wecom_enabled(settings.enabled)
    body = await request.body()

    try:
        if settings.encrypted_callbacks_enabled:
            crypto = WeComMessageCrypto(
                token=settings.token,
                encoding_aes_key=settings.encoding_aes_key,
                corp_id=settings.corp_id,
            )
            plaintext_xml = crypto.decrypt_message(
                msg_signature=msg_signature,
                timestamp=timestamp,
                nonce=nonce,
                body=body,
            )
        else:
            plaintext_xml = body.decode("utf-8")
            if not verify_signature(settings.token, msg_signature, timestamp, nonce, plaintext_xml):
                raise WeComCryptoError("Invalid WeCom callback signature.")

        message = WeComIncomingMessage.from_xml_fields(parse_xml_message(plaintext_xml))
        handler = build_wecom_message_handler()
        try:
            result = handler.handle_text_message(message)
        finally:
            close = getattr(handler, "close", None)
            if callable(close):
                close()

        if settings.response_mode == "active":
            return PlainTextResponse("success")

        reply_xml = build_text_reply_xml(
            to_user=message.from_user_name,
            from_user=message.to_user_name,
            content=result.reply.content,
            create_time=int(time.time()),
        )
        if settings.encrypted_callbacks_enabled:
            return PlainTextResponse(crypto.encrypt_message(reply_xml=reply_xml, timestamp=timestamp, nonce=nonce))
        return PlainTextResponse(reply_xml)
    except WeComCryptoError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except Exception:
        logger.exception("wecom_callback_failed")
        return PlainTextResponse("success")


@router.get("/integrations/wecom/oauth/callback")
def wecom_oauth_callback(code: str = Query(...)):
    settings = wecom_settings_from_app_settings(get_settings())
    _ensure_wecom_enabled(settings.enabled)
    client = WeComClient(settings)
    try:
        return client.get_user_info_by_code(code)
    finally:
        client.close()


def build_wecom_message_handler() -> WeComMessageHandler:
    settings = wecom_settings_from_app_settings(get_settings())
    client = WeComClient(settings) if settings.response_mode == "active" else None
    return WeComMessageHandler(
        settings=settings,
        rag_service=build_rag_service(),
        user_mapping_store=JsonWeComUserMappingStore(settings.user_mapping_path),
        wecom_client=client,
    )


def _ensure_wecom_enabled(enabled: bool) -> None:
    if not enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="WeCom integration is disabled.")
