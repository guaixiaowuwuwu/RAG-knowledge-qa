import base64
import hashlib
import os
import struct
from secrets import compare_digest
from xml.etree import ElementTree


class WeComCryptoError(ValueError):
    pass


def calculate_signature(token: str, timestamp: str, nonce: str, encrypted: str) -> str:
    parts = [token, timestamp, nonce, encrypted]
    parts.sort()
    return hashlib.sha1("".join(parts).encode("utf-8")).hexdigest()


def verify_signature(token: str, signature: str, timestamp: str, nonce: str, encrypted: str) -> bool:
    expected = calculate_signature(token, timestamp, nonce, encrypted)
    return compare_digest(expected, signature or "")


class WeComMessageCrypto:
    def __init__(self, *, token: str, encoding_aes_key: str, corp_id: str):
        if not token:
            raise WeComCryptoError("WeCom token is required.")
        if len(encoding_aes_key or "") != 43:
            raise WeComCryptoError("WeCom EncodingAESKey must be 43 characters.")
        self.token = token
        self.encoding_aes_key = encoding_aes_key
        self.corp_id = corp_id
        self._aes_key = base64.b64decode(f"{encoding_aes_key}=", validate=True)
        if len(self._aes_key) != 32:
            raise WeComCryptoError("WeCom EncodingAESKey must decode to 32 bytes.")

    def verify_url(self, *, msg_signature: str, timestamp: str, nonce: str, echostr: str) -> str:
        self._assert_signature(msg_signature, timestamp, nonce, echostr)
        return self.decrypt(echostr)

    def decrypt_message(self, *, msg_signature: str, timestamp: str, nonce: str, body: str | bytes) -> str:
        encrypted = parse_encrypted_xml(body)
        self._assert_signature(msg_signature, timestamp, nonce, encrypted)
        return self.decrypt(encrypted)

    def encrypt_message(self, *, reply_xml: str, timestamp: str, nonce: str) -> str:
        encrypted = self.encrypt(reply_xml)
        signature = calculate_signature(self.token, timestamp, nonce, encrypted)
        return build_encrypted_xml(encrypted=encrypted, signature=signature, timestamp=timestamp, nonce=nonce)

    def encrypt(self, plaintext: str) -> str:
        payload = (
            os.urandom(16)
            + struct.pack("!I", len(plaintext.encode("utf-8")))
            + plaintext.encode("utf-8")
            + self.corp_id.encode("utf-8")
        )
        padded = _pkcs7_pad(payload)
        cipher = _build_aes_cbc_cipher(self._aes_key, self._aes_key[:16], encrypt=True)
        encrypted = cipher.update(padded) + cipher.finalize()
        return base64.b64encode(encrypted).decode("ascii")

    def decrypt(self, encrypted: str) -> str:
        try:
            ciphertext = base64.b64decode(encrypted, validate=True)
        except ValueError as exc:
            raise WeComCryptoError("Invalid base64 encrypted payload.") from exc

        cipher = _build_aes_cbc_cipher(self._aes_key, self._aes_key[:16], encrypt=False)
        padded = cipher.update(ciphertext) + cipher.finalize()
        payload = _pkcs7_unpad(padded)
        if len(payload) < 20:
            raise WeComCryptoError("Decrypted payload is too short.")

        message_length = struct.unpack("!I", payload[16:20])[0]
        message_start = 20
        message_end = message_start + message_length
        message = payload[message_start:message_end]
        corp_id = payload[message_end:].decode("utf-8", errors="replace")
        if self.corp_id and corp_id != self.corp_id:
            raise WeComCryptoError("CorpId mismatch in decrypted payload.")
        return message.decode("utf-8")

    def _assert_signature(self, msg_signature: str, timestamp: str, nonce: str, encrypted: str) -> None:
        if not verify_signature(self.token, msg_signature, timestamp, nonce, encrypted):
            raise WeComCryptoError("Invalid WeCom callback signature.")


def parse_encrypted_xml(body: str | bytes) -> str:
    root = _parse_xml(body)
    encrypted = root.findtext("Encrypt")
    if not encrypted:
        raise WeComCryptoError("Encrypted WeCom XML is missing Encrypt.")
    return encrypted


def parse_xml_message(body: str | bytes) -> dict[str, str]:
    root = _parse_xml(body)
    return {child.tag: child.text or "" for child in root}


def build_text_reply_xml(*, to_user: str, from_user: str, content: str, create_time: int) -> str:
    return (
        "<xml>"
        f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
        f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
        f"<CreateTime>{create_time}</CreateTime>"
        "<MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{_cdata_safe(content)}]]></Content>"
        "</xml>"
    )


def build_encrypted_xml(*, encrypted: str, signature: str, timestamp: str, nonce: str) -> str:
    return (
        "<xml>"
        f"<Encrypt><![CDATA[{encrypted}]]></Encrypt>"
        f"<MsgSignature><![CDATA[{signature}]]></MsgSignature>"
        f"<TimeStamp>{timestamp}</TimeStamp>"
        f"<Nonce><![CDATA[{nonce}]]></Nonce>"
        "</xml>"
    )


def _parse_xml(body: str | bytes) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        raise WeComCryptoError("Invalid WeCom XML payload.") from exc


def _pkcs7_pad(data: bytes, block_size: int = 32) -> bytes:
    padding = block_size - (len(data) % block_size)
    return data + bytes([padding]) * padding


def _pkcs7_unpad(data: bytes, block_size: int = 32) -> bytes:
    if not data:
        raise WeComCryptoError("Empty decrypted payload.")
    padding = data[-1]
    if padding < 1 or padding > block_size or padding > len(data):
        raise WeComCryptoError("Invalid PKCS#7 padding.")
    if data[-padding:] != bytes([padding]) * padding:
        raise WeComCryptoError("Invalid PKCS#7 padding bytes.")
    return data[:-padding]


def _build_aes_cbc_cipher(key: bytes, iv: bytes, *, encrypt: bool):
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ModuleNotFoundError as exc:
        raise WeComCryptoError("The cryptography package is required for WeCom AES callbacks.") from exc

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    return cipher.encryptor() if encrypt else cipher.decryptor()


def _cdata_safe(value: str) -> str:
    return value.replace("]]>", "]]]]><![CDATA[>")
