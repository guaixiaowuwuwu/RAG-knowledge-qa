from dataclasses import dataclass


@dataclass(frozen=True)
class WeComIncomingMessage:
    to_user_name: str
    from_user_name: str
    create_time: int
    msg_type: str
    content: str = ""
    msg_id: str | None = None

    @classmethod
    def from_xml_fields(cls, fields: dict[str, str]) -> "WeComIncomingMessage":
        create_time = fields.get("CreateTime") or "0"
        try:
            parsed_create_time = int(create_time)
        except ValueError:
            parsed_create_time = 0
        return cls(
            to_user_name=fields.get("ToUserName", ""),
            from_user_name=fields.get("FromUserName", ""),
            create_time=parsed_create_time,
            msg_type=fields.get("MsgType", ""),
            content=(fields.get("Content") or "").strip(),
            msg_id=fields.get("MsgId") or None,
        )


@dataclass(frozen=True)
class WeComReply:
    content: str
    send_active_message: bool = False
    card_url: str | None = None


@dataclass(frozen=True)
class WeComUserMapping:
    tenant_id: str
    wecom_userid: str
    system_user_id: str
    display_name: str | None = None
    department_ids: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    permission_version: str = "local-v1"
