import pytest

from app.integrations.wecom.crypto import WeComMessageCrypto, calculate_signature, verify_signature


TOKEN = "test-token"
CORP_ID = "wxcorp123"
AES_KEY = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"


def test_calculates_wecom_sha1_signature_from_sorted_parts():
    signature = calculate_signature(TOKEN, "1700000000", "nonce", "encrypted")

    assert signature == "ee4446ae8c59be3373fa6162e674fe561f9cbc93"
    assert verify_signature(TOKEN, signature, "1700000000", "nonce", "encrypted") is True


def test_encrypted_url_verification_decrypts_echostr():
    crypto = WeComMessageCrypto(token=TOKEN, encoding_aes_key=AES_KEY, corp_id=CORP_ID)
    echostr = crypto.encrypt("hello-wecom")
    signature = calculate_signature(TOKEN, "1700000001", "nonce-1", echostr)

    assert crypto.verify_url(
        msg_signature=signature,
        timestamp="1700000001",
        nonce="nonce-1",
        echostr=echostr,
    ) == "hello-wecom"


def test_encrypted_message_rejects_invalid_signature():
    crypto = WeComMessageCrypto(token=TOKEN, encoding_aes_key=AES_KEY, corp_id=CORP_ID)
    encrypted = crypto.encrypt("<xml><Content><![CDATA[hi]]></Content></xml>")
    body = f"<xml><Encrypt><![CDATA[{encrypted}]]></Encrypt></xml>"

    with pytest.raises(ValueError, match="Invalid WeCom callback signature"):
        crypto.decrypt_message(
            msg_signature="bad",
            timestamp="1700000002",
            nonce="nonce-2",
            body=body,
        )
