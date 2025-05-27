# citer2_public
Repository for citer2 core codes

本システムは「FastAPI＋マルチエージェントPipeline」による文献検索プラットフォームです。

main.py：HTTP受付・認証・ジョブ管理・ファイル配信
pipeline.py：文分割→PubMed検索→LLMスコアリング→引用生成


main.py の役割:
FastAPI でエンドポイントを定義（同期GET／POST、SSEストリーム、ジョブ方式）
ユーザー認証（JWT＋Google OAuth＋有料プラン判定）
バックグラウンドタスク＆キューで進捗管理（janus／asyncio.Queue）
一時ファイル（RIS/JSON/CSV）生成＆ダウンロード提供

pipeline.py の役割:
Entrez（PubMed）＆PMC API 呼び出し
OpenAI ChatGPT を用いた「キーワード抽出」「文献支持度スコアリング」「上位選定」
SubordinateAgent＋EditAgent クラスで各機能を責務分割
ThreadPoolExecutor で文単位の並列実行


SubordinateAgent.analyze_sentence:
split_sentences：句読点・略語処理で文章を文単位に分割
extract_keywords＋build_queries：生物医系キーワード＋AND検索クエリ生成
find_citable_references：Entrez.esearch→efetch→ai_parallel（GPT-4miniによる5段階評価）
select_refs_with_llm：スコア>=閾値のプールからLLMに最終選定依頼


EditAgent＋レポート出力:
insert_and_export_endnote：本文へ EndNote タグ挿入＆RIS フォーマット生成
write_report/write_detail_report：CSV/JSON でsummary＋詳細レポートを一時出力
最終的に引用付きテキスト・RIS文字列・レポートID・referencesメタを返却


技術的工夫:
ジョブID＋SSE でフロントにリアルタイム進捗通知（2025/May/27現在は実装されてません）
asyncio＋run_in_executor でCPU＋IOバウンドを両立
Entrez APIキー管理＋再試行・バックオフ・スリープで429抑制
一時ディレクトリ活用で状態レスなスケーラビリティ


