from app.security.acl import DocumentACL, can_access_document
from app.security.context import RequestContext


def test_acl_blocks_cross_tenant_access():
    context = RequestContext(
        tenant_id="tenant-a",
        user_id="user-1",
        display_name="User",
        department_ids=("sales",),
        roles=("employee",),
        permission_version="v1",
        source="wecom",
    )
    acl = DocumentACL(tenant_id="tenant-b", is_public=True)

    assert can_access_document(context, acl) is False


def test_acl_allows_explicit_user_department_role_and_public_access():
    base_context = RequestContext(
        tenant_id="tenant-a",
        user_id="user-1",
        display_name="User",
        department_ids=("sales",),
        roles=("employee",),
        permission_version="v1",
        source="wecom",
    )

    assert can_access_document(
        base_context,
        DocumentACL(tenant_id="tenant-a", allowed_user_ids=("user-1",)),
    )
    assert can_access_document(
        base_context,
        DocumentACL(tenant_id="tenant-a", allowed_department_ids=("sales",)),
    )
    assert can_access_document(
        base_context,
        DocumentACL(tenant_id="tenant-a", allowed_roles=("employee",)),
    )
    assert can_access_document(
        base_context,
        DocumentACL(tenant_id="tenant-a", is_public=True),
    )
