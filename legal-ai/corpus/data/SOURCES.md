# Corpus Sources — TIP-CORPUS-001

- **Dataset:** `th1nhng0/vietnamese-legal-documents` (HuggingFace)
- **License:** CC BY 4.0
- **Origin:** vbpl.vn (Cổng thông tin điện tử Bộ Tư pháp)
- **DOI:** 10.57967/hf/8598
- **Pinned revision:** `0a39ad7eae8e6c188cb225c4b1443c3b346461d8`
- **Downloaded / built:** 2026-06-06
- **Citation name variant:** `loai_title`

## Article counts

| Tier | meta rows | kept (filtered) | docs parsed | articles |
|------|-----------|-----------------|-------------|----------|
| 0 (original, preferred) | — | — | — | 1044 |
| A (core, HTML)    | 153420 | 7287 | 6486 | 56441 |
| B (recall net)    | 518601 | 7358 | 7341 | 56023 |

- **Total articles (post-dedup):** 113508
- **Dedup dropped (tier preference 0 > A > B):** 45712
- **Docs skipped (no content / scan-only PDF):** 0

## Doc-type mapping (raw → kept/dropped)

| tier:raw_type | decision |
|---------------|----------|
| A:Bản ghi nhớ | DROP |
| A:Bộ luật | KEEP→Bộ luật |
| A:Chương trình | DROP |
| A:Chỉ thị | DROP |
| A:Công văn | DROP |
| A:Công ước | DROP |
| A:Hiến pháp | KEEP→Hiến pháp |
| A:Hiệp định | DROP |
| A:Luật | KEEP→Luật |
| A:Lệnh | DROP |
| A:Nghị Quyết | KEEP→Nghị quyết |
| A:Nghị quyết | KEEP→Nghị quyết |
| A:Nghị quyết liên tịch | KEEP→Nghị quyết liên tịch |
| A:Nghị định | KEEP→Nghị định |
| A:Nghị định thư | DROP |
| A:Pháp lệnh | KEEP→Pháp lệnh |
| A:Quyết định | DROP |
| A:Sắc lệnh | DROP |
| A:Thông báo | DROP |
| A:Thông tư | KEEP→Thông tư |
| A:Thông tư liên bộ | KEEP→Thông tư liên bộ |
| A:Thông tư liên tịch | KEEP→Thông tư liên tịch |
| A:Thỏa thuận | DROP |
| A:Văn bản hợp nhất | KEEP→Văn bản hợp nhất |
| A:Văn bản khác | DROP |
| A:Văn bản liên quan | DROP |
| B:Agreement | DROP |
| B:Announcement | DROP |
| B:Báo cáo | DROP |
| B:Circular | KEEP→Thông tư |
| B:Constitution | KEEP→Hiến pháp |
| B:Convention | DROP |
| B:Decision | DROP |
| B:Decree of Government | KEEP→Nghị định |
| B:Decree-Order | DROP |
| B:Directive | DROP |
| B:Documents about WTO | DROP |
| B:Instruction | DROP |
| B:Integrated document | KEEP→Văn bản hợp nhất |
| B:Joint circular | KEEP→Thông tư liên tịch |
| B:Kế hoạch | DROP |
| B:Law | KEEP→Luật |
| B:National Treaty | DROP |
| B:Official Dispatch | DROP |
| B:Official Telegram | DROP |
| B:Order | DROP |
| B:Ordinance | KEEP→Pháp lệnh |
| B:Other Documents | DROP |
| B:Protocal | DROP |
| B:Regulation | DROP |
| B:Resolution | KEEP→Nghị quyết |
| B:Standing Rules | DROP |
| B:Statute | DROP |
| B:Tiêu chuẩn Việt Nam | DROP |
| B:Tiêu chuẩn XDVN | DROP |
| B:Treaty | DROP |
| B:WTO Documents | DROP |
| B:WTO_Undertaken by Vietnam | DROP |

## Filters applied

- **Effect keep:** A=['Còn hiệu lực', 'Hết hiệu lực một phần'], B=['In effect']
- **Doc-type keep (VI):** ['Bộ luật', 'Luật', 'Pháp lệnh', 'Hiến pháp', 'Nghị định', 'Nghị quyết', 'Nghị quyết liên tịch', 'Thông tư', 'Thông tư liên tịch', 'Thông tư liên bộ', 'Văn bản hợp nhất']
- **Domain filter applied:** {'A': True, 'B': True}
- **Domain keywords:** 82 terms (title + sectors substring match; see hf_config.json)
