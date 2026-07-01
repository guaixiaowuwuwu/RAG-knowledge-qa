import logging
from dataclasses import dataclass

from app.core.config import get_settings
from app.integrations.wecom.config import WeComSettings
from app.integrations.wecom.schemas import WeComIncomingMessage, WeComReply, WeComUserMapping
from app.integrations.wecom.user_mapping import WeComUserMappingStore
from app.rag.service import Answer, RagService
from app.security.context import RequestContext


logger = logging.getLogger(__name__)

MAX_WECOM_TEXT_CHARS = 1800


@dataclass(frozen=True)
class WeComHandlerResult:
    reply: WeComReply
    context: RequestContext
    answer: Answer | None = None


class WeComMessageHandler:
    def __init__(
        self,
        *,
        settings: WeComSettings,
        rag_service: RagService,
        user_mapping_store: WeComUserMappingStore,
        wecom_client: object | None = None,
    ):
        self.settings = settings
        self.rag_service = rag_service
        self.user_mapping_store = user_mapping_store
        self.wecom_client = wecom_client

    def handle_text_message(self, message: WeComIncomingMessage) -> WeComHandlerResult:
        context = self.context_for_user(message.from_user_name)
        if message.msg_type != "text" or not message.content:
            reply = WeComReply(content="当前仅支持文本问题。")
            self._maybe_send_active(message.from_user_name, reply)
            return WeComHandlerResult(reply=reply, context=context)

        try:
            answer = self.rag_service.answer(
                question=message.content,
                top_k=getattr(get_settings(), "retrieval_top_k", 4),
                context=context,
            )
            content = format_wecom_answer(answer)
            reply = WeComReply(
                content=content,
                send_active_message=self.settings.response_mode == "active",
            )
        except Exception:
            logger.exception("wecom_rag_answer_failed user=%s", message.from_user_name)
            reply = WeComReply(content="服务暂时不可用，请稍后重试。")
            answer = None

        self._maybe_send_active(message.from_user_name, reply)
        return WeComHandlerResult(reply=reply, context=context, answer=answer)

    def context_for_user(self, wecom_userid: str) -> RequestContext:
        mapping = self.user_mapping_store.get_by_wecom_userid(wecom_userid)
        if mapping is None:
            settings = get_settings()
            mapping = WeComUserMapping(
                tenant_id=str(getattr(settings, "default_tenant_id", "default")),
                wecom_userid=wecom_userid,
                system_user_id=f"wecom:{wecom_userid}",
                display_name=wecom_userid,
                department_ids=(),
                roles=(),
                permission_version=str(getattr(settings, "permission_version", "local-v1")),
            )
        return RequestContext(
            tenant_id=mapping.tenant_id,
            user_id=mapping.system_user_id,
            display_name=mapping.display_name,
            department_ids=mapping.department_ids,
            roles=mapping.roles,
            permission_version=mapping.permission_version,
            source="wecom",
        )

    def _maybe_send_active(self, touser: str, reply: WeComReply) -> None:
        if self.settings.response_mode != "active" or self.wecom_client is None:
            return
        try:
            send_text = getattr(self.wecom_client, "send_text")
            send_text(touser=touser, content=reply.content)
        except Exception:
            logger.exception("wecom_active_reply_failed user=%s", touser)

    def close(self) -> None:
        close = getattr(self.wecom_client, "close", None)
        if callable(close):
            close()


def format_wecom_answer(answer: Answer, *, max_chars: int = MAX_WECOM_TEXT_CHARS) -> str:
    content = answer.answer.strip()
    citations = []
    for index, source in enumerate(answer.sources[:3], start=1):
        page = f" p.{source.page}" if source.page is not None else ""
        citations.append(f"[{index}] {source.source}{page}")
    if citations:
        content = f"{content}\n\n引用：\n" + "\n".join(citations)
    if len(content) <= max_chars:
        return content
    suffix = "\n\n回答较长，已截断。请在 Web 端查看完整结果。"
    return content[: max_chars - len(suffix)].rstrip() + suffix
