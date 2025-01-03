# MEMO


- FastAPIの起動
```bash
uv run fastapi dev
```


## Azure関係

### RAGとは？
- 大規模援護モデルが知らない情報を外部の検索システムで検索し、その検索結果を情報源として回答する手法のこと
- 用語
  - グラウディング
  　大規模言語モデルに外部情報を連携することをグラウディングと呼ぶ
  - オーケストレータ
    ユーザーの質問を受けて関連するドキュメントを検索したり、大規模言語モデルに回答作成の指示を与えたりするモジュールはオーケストレータと呼ぶ
- メリット
  - モデルが最新活信頼性の高い情報にアクセスできること
  - ユーザーがモデルの参照情報源を認識でき、生成結果の正確さを確認・検証できること。⇒ハルシネーション対策
- 処理フロー
  - step1. ユーザーが質問
    - {question}及び過去のチャット履歴{chat_history}をコンテキストとして検索クエリを生成させる
  - step2. 検索クエリを生成
    - LLMがAzureAISearch用に{question}を生成する。
  - step3. ドキュメント検索
    - {sources}内の検索結果及び過去のチャット履歴{chat_history}をコンテキストとして回答を生成させる
    - チャンク分割された上位3件分をpickupする。
  - step4. 回答生成
    - 情報を統合して回答（プロンプトの中で参照元のファイル名を引用して出力するように指示）
    - さらにフォローアップ質問を生成して対話を加速する。
- システム構成(Azure PaaSで実現するRAGアーキテクチャ)
  - 使用するAzureサービス
    - 1. AzureOpenAIService
      - 使用目的
        検索クエリの生成と回答の生成にgptモデルを、Embeddings生成にtext-embedding-adaモデルを使用
        ※どちらも30KTPMのクウォータの空きが必要
      - プランとスペック
        Standard S0プラン
      - 料金体系
        使用した1000トークンあたりの課金、1門当たり1000トークンが使用される。
        簡単なテストであれば100円/日程度
    - 2. AzureAppService
      - 使用目的
        チャットUIアプリケーションのホスティングに使用
      - プランとスペック
        Basic B1プラン。1CPUコア,1.75GB RAM
      - 料金体系
        1時間あたりの従量課金、3円/1時間。
    - 3. Azure AI Document Intelligence
      - 使用目的
        PDFの読み取りと構造化を高精度に行う。事前構築済みレイアウトモデルを使用？
      - プランとスペック
        Standard S0プラン
      - 料金体系
        100ページで約146円。
    - 4. Azure AI Search
      - 使用目的
        検索インデックス及びベクトルインデックスとして使用
      - プランとスペック
        Basicプラン
      - 料金体系
        1レプリカ、無料レベルのセマンティック検索。約20円/時間
    - 5. Azure Blob Storage
      - 使用目的
        PDFファイルの保管場所として使用
      - プランとスペック
        Standard LRS(ローカル冗長)
      - 料金体系
        従量課金、ストレージと読み取り操作ごとの価格。10円/月。
    - 6. Azure Cosmos DB
      - 使用目的
        チャット履歴の保管場所として使用。本家ChatGPTのサービスにも使用されている。
      - プランとスペック
        標準プロビジョ二ング済みスループット、400RU/s
      - 料金体系
        従量課金、約6円/時間。

- 情報源
  - [RAG論文](https://arxiv.org/abs/2005.11401)
  - [GitHubサンプルリポジトリ](https://github.com/shohei1029/book-azureopenai-sample)
  - [AzureDemo](https://github.com/Azure-Samples/azure-search-openai-demo)
  - [AzureDemoPostgreSQL](https://github.com/pamelafox/rag-on-postgres)

- 検索システム
  - インターネット検索
  - ドキュメント検索(ベクトル検索)
  - データベース検索(RDB)

- 注意事項
  - ドキュメントの文字数について
  - 文章検索の場合

### Azure AI Searchとは
- 社内文書を格納するナレッジベースとしてはAzureAISearchを使用
- PaaS
- Azure AI Searchを使えばドキュメント内容をindexとして登録しておき、検索時はインデックスからドキュメントを引き当てるので高速に検索することができる。
  - 処理フロー
    - step1. インデックス作成
    - step2. ドキュメント検索
- メリット
  - 多様なデータソースとフォーマットの対応
    - PDFやテーブルデータ、JSONなどの多種多様なデータフォーマットに対応している。
    - Blob StorageやAzure CosmosDB等様々なデータソースにも対応
  - スケーラビリティ
    インデックスのパーティション数やレプリカ数を調整し、負荷に応じて柔軟にスケールアウト可能
  - 豊富な検索機能
  　フルテキスト検索に加え、Bingの検索エンジンで使用されるAI搭載のセマンティック検索やベクトル検索にも対応

#### インデックス作成の流れ
- インデックス作成の方法には2つ
  - 1. インデクサーの利用
    - Azure Blob StorageなどのAzure AI Searchが対応しているデータソースを指定する。
    - インデクサーは定期的にデータソースをスキャンし、インデックスを更新する。（Pull型）
    - PDFをテキストに変換する処理も内部で自動的に実行される。
  - 2. APIの利用
    - インデックスのレコード内容を直接APIリクエストのボディとして送信し、インデックスを作成する。（Push型）
    - ドキュメントからテキストを抽出する処理はユーザー側で行う必要はあるが、Pull型に比べてリアルタイムなインデックスの更新が可能
- インデックス付与について
  - json,csvは各行や項目が1レコードとして登録
  - その他Word、pdf,Excel,PowerPoint等は1ドキュメントに対して1レコードのインデックスが作成される。
- スキルセットについて
  インデクサーによって取得したテキストに対し、スキルセットを活用することでキーフレーズ抽出などさまざまな処理を行うことが可能
  - カスタムエンティティの参照
  - キーフレーズ抽出
  - エンティティの認識(v3)
  - PII(個人情報)検出
  - テキスト分割
  - テキスト翻訳
  - 画像分析
- アナライザーについて
  - 検索対象のフィールドは、アナライザーによってテキストから単語に分割されてからインデックスに保存される。
  - アナライザーはインデックス作成時に利用されるだけでなく、クエリ実行時にも利用される。
  - アナライザーは単に単語の分割だけでなく、以下の処理も行う。
    - ストップワードの削除
    - フレーズやハイフン付きの単語を個々の要素に分解
    - 大文字の単語を小文字に変換
    - 単語をその基本形に簡略化し、異なる時制でも一致を容易にする
  - 日本語ではja.lucene OR ja.microsoft
- インデックスのスキーマ
  - key
    インデックス内のドキュメントの一意識別子
  - searchable
    フルテキスト検索可能
  - filterable
    フィルタークエリで利用可能
  - sortable
    デフォルトでは結果スコアで並べ替えるか、フィールドに基づいて並べ替えが可能
  - facetable
    カテゴリ別のヒット数として検索可能に含めることが可能
  - retrievable
  　検索結果に含めか否か
- チャンク分割
- ドキュメント検索
  - フルテキスト検索（BM25アルゴリズム）
    - step1. ユーザーがクエリを発行すると、クエリパーサーが構文を解析、検索文を抜き出す
    - step2. アナライザーが検索文を単語に分解
    - step3. 検索を実行し、検索スコアに基づいて結果をソート
  - ベクトル検索
    - Embeddingを活用したベクトル検索
    - 類似度計算にはcos類似度が一般的に使用される
    - 検索アルゴリズム
      基本的には1.を利用し、検索精度が悪くて小規模データの場合は2.に切り替える。
      - 1. HNSW(Hierarchi Navigative Small World) 
        階層グラフ構造により高速でスケーラブルな検索を実現。検索精度と計算コストのトレードオフを調整
      - 2. KNN(Exhaustive K-nearest neighbors)
        全てのデータ点の類似度を計算。
  - セマンティック検索
  　フルテキストで検索された結果に対し、独自のAIモデルによって関連する結果を並び替える手法
  - ハイブリッド検索
  　ハイブリッド検索とセマンティックランク付けの検索手法はほとんどのクエリでかなり高い精度の検索結果を実現されていることが報告されている。

### Azure AI Searchの設定
- [参考](https://learn.microsoft.com/ja-jp/azure/search/search-create-service-portal)
- [確認]AzureAISearchの料金について　Free版でどこまでできるのか？有料版との差異は？
- [サンプルコード](https://github.com/shohei1029/book-azureopenai-sample/tree/main/aoai-rag/notebooks)
- [Avanadeサンプルコード](https://github.com/AvanadeJapanPublishingQuery/AzureOpenAIServicePracticalGuide-book)
- [確認]Storageアカウント及びBlobstorage設定

- アナライザーja.luceneについて
　ja.luceneというスタンダードな日本語アナライザーに搭載されている辞書ベースのトークナイザーによって、これらのトークンに分解される。このトークンを用いて転置インデックスが構築される。

### Azure AI Document Intelligenceの設定
- [Documents Intelligence](https://learn.microsoft.com/ja-jp/azure/ai-services/document-intelligence/prebuilt/layout?view=doc-intel-4.0.0&tabs=sample-code)


## Azure App Serviceへのデプロイ
- ローカル開発環境にて一部変更
  - app/ フロントエンド/バックエンド
    ```powershell
    azd deploy
    ```
  - infra/ azure.yaml インフラ関係
    ```powershell
    azd up
    ```
- 環境設定ファイルの変更
  自動構築された環境名と環境設定は.azureディレクトリ配下に保存される。
  もしAzureOpenAIServiceAPIやAzureAISearchの設定を変更したい場合は.azureディレクトリを削除する。
- 追加のドキュメントをインデックス化
  dataフォルダに入れて、scripts/prepdocs.shまたはscripts/prepdocs.ps1を実行する。

## チャットUI実装の注意事項
- アクセスコントロール機能実装
  ユーザーの識別子をAzureAISearchのインデックスフィールドに格納し、フィルタクエリで出しわける手法を採用
  [セキュリティフィルター](https://learn.microsoft.com/ja-jp/azure/search/search-security-trimming-for-azure-search)
- チャンク分割の重要性
  - text-embedding-ada-002モデルの入力テキスト最大長は8191トークン
  - Microsoftの定量評価だとチャンク間の重複を25%,512トークン毎に分割すると良い結果が出ることが判明。
- データインジェストの自動化
  - AzureAISearchにはデータインジェストを自動化するインデクサー機能がある。
  - カスタムスキル機能の利用
  - ファイルの取り込み⇒インデックス化⇒チャンク化⇒Embedding化⇒チャンクのインデックス化までの一連の流れを以下のソースコードで実装
    - [Azure OpenAI Embeddings Generator Skill](https://learn.microsoft.com/ja-jp/azure/search/cognitive-search-skill-azure-openai-embedding)
    - [azure-search-vector-ingestion-python-sample](https://github.com/Azure/azure-search-vector-samples/blob/main/demo-python/code/integrated-vectorization/readme.md)
    ※Azure AI Searchの機能にドキュメントのチャンク機能がある。よってまずはAzure Portalで検証してみてそれでもユーザーのニーズを満たせない場合はコードベースの実装を行う。
- RAGの評価
  - 課題
    - 1. 検索精度の問題
      - ユーザーの質問に関連するドキュメントを検索できない場合
    - 2. 生成精度の問題
      - プロンプトの書き方に不備があったり、複雑な文脈の理解が求められたりする場合、回答精度が悪くなる可能性がある。
  - RAGの生成精度評価
    - 1. 関連性の評価
    - 2. 一貫性の評価
    - 3. 類似性の評価
    - 4. コンテキストの適合率と再現率
- RAGの回答精度向上のTips
  - 1. 検索システムのインデックスをユースケース毎作成
  - 2. 検索時、フィルタ機能を活用することも効果的（ユーザーからの質問に対して、質問がどのカテゴリに属するか判定させるロジック
  構築する。

## Copilot stack
- 第1層：Copilotフロントエンド
- 第2層：AIオーケストレーション
  - エージェントの実装(ReAct)
    タスクの処理フローをGPTが自動的に決めてさまざまなツールと連携することで回答を生成する手法。
    - step1. 利用するツールを定義
    - step2. 回答のフォーマットを設定
    - step3. 質問とツール定義に基づいてどのような行動をとればよいかを思考
    - step4. ツールの実行と結果の取得
    - step5. 結果を取得したうえでどのような行動を取ればよいか思考
    - step6. 思考の結果、目的を達成していれば最終的な回答を生成
  - Planning&Execution（計画と実行）
    Reactとは異なりアクションを事前に定義しておき、複雑な問題に対して一貫性のある推論を目指すアプローチ
    - step1.LLMを使用して明確な手順で指示に答えるための計画を作成（プランナー）
    - step2.事前に準備されたツールを実行して各ステップを解決（エグゼキューター）
    ※Langchainであれば「Plan-and-Excecuteエージェント」を活用
  - Plugin
    - 独自Copilotの開発のアーキテクチャと実装
    - ChatGPTプラグインの実装
      - Function calling
        LangchainではAgentType.OPENAI_FUNCTIONSと指定するだけで利用できる
      - プラグインのサンプルリポジトリ
        - [ChatGPT plugins quickstart](https://github.com/openai/plugins-quickstart)
        - [ChatGPT Plugin Quickstart using Python and FastAPI](https://github.com/Azure-Samples/openai-plugin-fastapi)
- 第3層：基盤モデル
- 第4層：AIインフラストラクチャ