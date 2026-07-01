# 企业微信集成说明

核对日期：2026-06-22。

已核对的企业微信官方文档：

- 回调配置：<https://developer.work.weixin.qq.com/document/path/90930>
- 获取 access_token：<https://developer.work.weixin.qq.com/document/path/91039>
- 网页授权获取访问用户身份：<https://developer.work.weixin.qq.com/document/path/91023>
- 发送应用消息：<https://developer.work.weixin.qq.com/document/path/90238>

## 当前实现范围

当前仓库已提供企业微信自建应用的 pilot 集成，并接入同一套身份、ACL、审计和观测链路：

- `GET /integrations/wecom/callback`：用于企业微信后台 URL 验证。
- `POST /integrations/wecom/callback`：接收文本消息回调，验证签名，解密消息，构造 `RequestContext(source="wecom")`，调用 RAG，并返回被动回复或主动发送。
- `GET /integrations/wecom/oauth/callback`：保留网页登录 OAuth 回调入口，返回企业微信用户身份响应。
- `JsonWeComUserMappingStore`：从 JSON 文件加载 `wecom_userid` 到系统用户、部门、角色和权限版本的映射。
- `WeComClient`：获取并缓存 `access_token`，调用获取用户身份、读取成员、发送文本和 textcard 应用消息接口。
- 企业微信文本问题会写入问答审计；返回来源已经经过 tenant、用户、部门、角色和权限版本过滤。

这仍不是生产 SLA 证明。当前实现适合企业试点和 staging smoke；真实生产还需要入口限流、集中日志告警、审计库迁移、密钥系统、备份恢复和压测记录。

## 环境变量

```dotenv
WECOM_ENABLED=true
WECOM_CORP_ID=wwxxxxxxxxxxxx
WECOM_AGENT_ID=1000001
WECOM_SECRET=replace-with-app-secret
WECOM_TOKEN=replace-with-callback-token
WECOM_ENCODING_AES_KEY=replace-with-43-char-encoding-aes-key
WECOM_CALLBACK_PATH=/integrations/wecom/callback
WECOM_RESPONSE_MODE=active
WECOM_USER_MAPPING_PATH=data/runtime/wecom_users.json
```

企业或 staging 环境还应同时设置：

```dotenv
AUTH_ENABLED=true
ADMIN_API_KEYS=replace-with-random-admin-key
DEFAULT_TENANT_ID=default
PERMISSION_VERSION=staging-perm-v1
DOCUMENTS_MANIFEST_PATH=data/documents_manifest.json
AUDIT_DB_PATH=data/runtime/audit.sqlite3
INDEX_ROOT_DIR=data/indexes
ACTIVE_INDEX_VERSION_PATH=data/indexes/active_version.txt
INGESTION_MODE=async
```

`WECOM_RESPONSE_MODE`：

- `active`：回调处理后返回 `success`，通过发送应用消息接口主动回复用户。适合答案较长或需要后续扩展卡片。
- `passive`：直接在回调响应中返回文本 XML；启用回调加密时会返回加密 XML。

## 用户映射文件

示例 `data/runtime/wecom_users.json`：

```json
{
  "users": [
    {
      "tenant_id": "default",
      "wecom_userid": "alice",
      "system_user_id": "user-alice",
      "display_name": "Alice",
      "department_ids": ["finance"],
      "roles": ["employee"],
      "permission_version": "local-v1"
    }
  ]
}
```

未命中的企业微信用户会被视为已认证用户，但没有部门和角色权限，只能访问同租户 public 文档。

## 企业微信后台配置

1. 在企业微信管理后台创建自建应用，记录 `CorpId`、`AgentId`、`Secret`。
2. 在应用的接收消息配置中填写 URL，例如 `https://staging.example.com/integrations/wecom/callback`。
3. 配置 Token 和 EncodingAESKey，并同步写入 `.env`。
4. 保存配置时，企业微信会调用 `GET /integrations/wecom/callback`，系统会验证签名并返回解密后的 `echostr`。
5. 为应用配置可见范围，确保企业微信侧和本系统 ACL 映射一致。

## 本地和 staging 验证

本地可用内网穿透把 `127.0.0.1:8000` 暴露给企业微信后台。启动前确认：

```bash
AUTH_ENABLED=true WECOM_ENABLED=true make dev
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/metrics
```

离线单元测试：

```bash
.venv/bin/pytest tests/test_wecom_signature.py tests/test_wecom_routes.py tests/test_wecom_handlers.py -v
```

企业链路 smoke：

```bash
.venv/bin/pytest tests/test_enterprise_smoke.py -v
```

后台保存回调 URL 后，用企业微信的 URL 验证请求确认 `GET /integrations/wecom/callback` 成功；再发送一条文本消息，检查：

- 无效签名返回 403，且不会调用 RAG。
- 有效文本消息生成 `source="wecom"` 的审计 session。
- 返回来源只包含该企业微信用户 ACL 允许访问的文档。
- 日志里只有 request id、tenant、user hash 和 index version，没有 `WECOM_SECRET`、`WECOM_TOKEN`、AES key 或加密正文。

## 常见问题

- 签名不匹配：检查 Token、`msg_signature`、`timestamp`、`nonce`、密文是否按字典序拼接后 SHA1；不要把明文 XML 用于加密回调签名。
- 解密失败：检查 EncodingAESKey 是否为 43 位，解密后的 CorpId 是否等于 `WECOM_CORP_ID`。
- 用户只能看到 public 文档：检查 `WECOM_USER_MAPPING_PATH` 是否包含对应 `wecom_userid`，以及部门、角色是否和文档 ACL 匹配。
- 主动回复失败：检查 `WECOM_SECRET`、`WECOM_AGENT_ID`、应用可见范围，以及企业微信 API 返回的 `errcode`。
- 审计没有记录：检查 `AUDIT_DB_PATH` 是否可写，以及 RAG 调用是否成功完成或安全拒答。
- 回答引用不符合预期：先看 `data/documents_manifest.json` 的 pattern 是否匹配实际 source，再运行权限检索和企业 smoke 测试。
