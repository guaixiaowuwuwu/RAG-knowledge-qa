from types import SimpleNamespace

from app.integrations.wecom.config import WeComSettings
from app.integrations.wecom.handlers import WeComMessageHandler, format_wecom_answer
from app.integrations.wecom.schemas import WeComIncomingMessage, WeComUserMapping
from app.rag.service import Answer, Source


class FakeMappingStore:
    def __init__(self, mapping=None):
        self.mapping = mapping

    def get_by_wecom_userid(self, wecom_userid):
        return self.mapping


class FakeRagService:
    def __init__(self, answer):
        self.answer_payload = answer
        self.calls = []

    def answer(self, **kwargs):
        self.calls.append(kwargs)
        return self.answer_payload


class FakeWeComClient:
    def __init__(self):
        self.sent = []

    def send_text(self, **kwargs):
        self.sent.append(kwargs)
        return {"errcode": 0}


def test_text_message_builds_permission_context_and_calls_rag(monkeypatch):
    from app.integrations.wecom import handlers

    monkeypatch.setattr(handlers, "get_settings", lambda: SimpleNamespace(retrieval_top_k=3))
    mapping = WeComUserMapping(
        tenant_id="tenant-a",
        wecom_userid="alice",
        system_user_id="user-alice",
        display_name="Alice",
        department_ids=("finance",),
        roles=("analyst",),
        permission_version="perm-2",
    )
    rag_answer = Answer(
        answer="answer",
        sources=[Source(source="finance.md", page=2, chunk_index=0, content="source content")],
    )
    rag = FakeRagService(rag_answer)
    client = FakeWeComClient()
    handler = WeComMessageHandler(
        settings=WeComSettings(enabled=True, response_mode="active"),
        rag_service=rag,
        user_mapping_store=FakeMappingStore(mapping),
        wecom_client=client,
    )

    result = handler.handle_text_message(
        WeComIncomingMessage(
            to_user_name="agent",
            from_user_name="alice",
            create_time=1,
            msg_type="text",
            content="预算制度是什么？",
        )
    )

    context = rag.calls[0]["context"]
    assert context.source == "wecom"
    assert context.tenant_id == "tenant-a"
    assert context.user_id == "user-alice"
    assert context.department_ids == ("finance",)
    assert context.roles == ("analyst",)
    assert context.permission_version == "perm-2"
    assert "finance.md" in result.reply.content
    assert client.sent[0]["touser"] == "alice"
    assert client.sent[0]["content"] == result.reply.content


def test_unknown_user_gets_authenticated_public_only_context(monkeypatch):
    from app.integrations.wecom import handlers

    monkeypatch.setattr(
        handlers,
        "get_settings",
        lambda: SimpleNamespace(
            retrieval_top_k=4,
            default_tenant_id="default",
            permission_version="local-v1",
        ),
    )
    rag = FakeRagService(Answer(answer="public answer", sources=[]))
    handler = WeComMessageHandler(
        settings=WeComSettings(enabled=True, response_mode="passive"),
        rag_service=rag,
        user_mapping_store=FakeMappingStore(None),
    )

    result = handler.handle_text_message(
        WeComIncomingMessage(
            to_user_name="agent",
            from_user_name="unknown",
            create_time=1,
            msg_type="text",
            content="公共制度是什么？",
        )
    )

    assert result.context.user_id == "wecom:unknown"
    assert result.context.department_ids == ()
    assert result.context.roles == ()
    assert result.context.source == "wecom"


def test_long_answer_is_truncated_with_safe_suffix():
    answer = Answer(answer="A" * 2100, sources=[])

    content = format_wecom_answer(answer, max_chars=120)

    assert len(content) <= 120
    assert "已截断" in content


def test_wecom_upstream_failure_does_not_break_handler(monkeypatch):
    from app.integrations.wecom import handlers

    monkeypatch.setattr(handlers, "get_settings", lambda: SimpleNamespace(retrieval_top_k=4))

    class FailingClient:
        def send_text(self, **kwargs):
            raise RuntimeError("upstream failed")

    handler = WeComMessageHandler(
        settings=WeComSettings(enabled=True, response_mode="active"),
        rag_service=FakeRagService(Answer(answer="answer", sources=[])),
        user_mapping_store=FakeMappingStore(None),
        wecom_client=FailingClient(),
    )

    result = handler.handle_text_message(
        WeComIncomingMessage(
            to_user_name="agent",
            from_user_name="alice",
            create_time=1,
            msg_type="text",
            content="问题",
        )
    )

    assert result.reply.content == "answer"
