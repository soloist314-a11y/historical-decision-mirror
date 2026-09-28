# 源注册表

机器可读主注册表为 ../data/source_registry.json。

| source_id | 指定来源 | 发布边界 |
|---|---|---|
| ZZTJ_DRIVE_BAIYANG | 用户指定 Google Drive 柏杨版 EPUB，司马光正文为证据层，史家评论和译文分层 | 只发布注册表、完整性报告和必要锚点；完整电子书及解包正文留本地 |
| SHIJI_WIKISOURCE | 中文维基文库《史记》，精确篇章 URL 保留在 source_refs | 只发布来源和必要引用；不重新授权第三方文本 |

source_integrity.json 与 source-integrity-report.md 是继承的卷级完整性审计。本次重新下载指定 EPUB 并执行源锚点检查，不宣称本次重新实现了全部 294 卷的完整性审计。

本次 EPUB SHA-256：d349a3507a227224af2ae692c0ffe4e0f0cce8d051429d37118548f8cfd5c0a0。

69 个 ACTIVE 节点含资治通鉴来源，41 个含史记来源；由于部分节点跨段引用，合计 113 条来源引用。旧 SOURCE_VERIFIED 表示原作者校验记录；使用时仍须回读当前原文。
