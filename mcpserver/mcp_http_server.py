#!/usr/bin/env python3
"""LangConnect MCP Server using FastMCP"""

import json
import os
import sys
import base64
import time
import threading
from datetime import datetime
from getpass import getpass
from pathlib import Path
from typing import Optional

import httpx
import requests
from dotenv import load_dotenv
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from fastmcp import FastMCP

load_dotenv()


# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_REFRESH_TOKEN = os.getenv("SUPABASE_REFRESH_TOKEN", "")
HTTP_PORT = int(os.getenv("HTTP_PORT", "4200"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# Output parser for multi-query generation
class LineListOutputParser(BaseOutputParser[list[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> list[str]:
        # Split into lines, strip whitespace, and remove empties
        lines = [line.strip() for line in text.strip().split("\n")]
        return [line for line in lines if line]


# SUPABASE 계정을 통해 SUPABASE_JWT_SECRET을 생성하는 함수
def sign_in(email: str, password: str):
    """SUPABASE 등록된 이메일, 비밀번호를 통해 access_token, refresh_token을 발급받는 함수

    Args:
        email (str): SUPABASE 등록된 이메일
        password (str): SUPABASE 등록된 비밀번호

    Returns:
        access_token, refresh_token: SUPABASE_JWT_SECRET 키
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/signin",
            json={"email": email, "password": password},
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("access_token"), data.get("refresh_token")
        error = response.json()
        print(f"Sign in failed: {error.get('detail', 'Unknown error')}")
        return None, None

    except Exception as e:
        print(f"Error during sign in: {e!s}")
        return None, None


def test_token(token: str):
    """
    콜렉션을 조회해서 토큰이 잘 작동하는 지 테스트하는 함수
    정상일 시 200 반환

    Args:
        token (str): SUPABASE_JWT_SECRET 토큰 값
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/collections", headers={"Authorization": f"Bearer {token}"}
        )
        return response.status_code == 200
    except:
        return False


def update_env_file(access_token: str, refresh_token: Optional[str] = None):
    """SUPABASE_JWT_SECRET(액세스 토큰)과 SUPABASE_REFRESH_TOKEN(리프레시 토큰)을 .env 파일에 업데이트하는 함수
    .env가 없으면 생성하고, 있으면 업데이트

    Args:
        access_token (str): 액세스 토큰 값
        refresh_token (Optional[str]): 리프레시 토큰 값 (주어지면 저장)
    """
    env_path = Path(__file__).parent.parent / ".env"

    # .env 없을 시 생성
    if not env_path.exists():
        print("⚠️  .env file not found. Creating new one...")
        with open(env_path, "w") as f:
            f.write(f"SUPABASE_JWT_SECRET={access_token}\n")
            if refresh_token:
                f.write(f"SUPABASE_REFRESH_TOKEN={refresh_token}\n")
        return

    # 있을 시 존재하는 .env 파일 로드
    with open(env_path) as f:
        lines = f.readlines()

    # 값 업데이트 또는 없을 시 추가
    updated_access = False
    updated_refresh = False
    new_lines = []

    for line in lines:
        s = line.strip()
        if s.startswith("SUPABASE_JWT_SECRET="):
            new_lines.append(f"SUPABASE_JWT_SECRET={access_token}\n")
            updated_access = True
        elif s.startswith("SUPABASE_REFRESH_TOKEN="):
            # refresh_token이 주어진 경우에만 업데이트, 아니면 원본 유지
            if refresh_token:
                new_lines.append(f"SUPABASE_REFRESH_TOKEN={refresh_token}\n")
            else:
                new_lines.append(line)
            updated_refresh = True
        else:
            new_lines.append(line)

    # 없던 키들 추가
    if not updated_access:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.append(f"SUPABASE_JWT_SECRET={access_token}\n")
    if not updated_refresh and refresh_token:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        new_lines.append(f"SUPABASE_REFRESH_TOKEN={refresh_token}\n")

    # 저장
    with open(env_path, "w") as f:
        f.writelines(new_lines)

    print("✅ Updated .env file with new token(s)")


# SUPABASE 계정, 비밀번호 입력해서 토큰 생성하고 .env 저장하는 함수
def get_access_token():
    """유저 권한을 통해 SUPABASE 토큰을 생성하는 함수

    email, password를 입력받아서 access token을 발급받은 후, .env 파일에 업데이트

    Returns:
        access_token, refresh_token
    """
    print("\n🔐 Authentication Required")
    print("=" * 40)
    print("Please sign in to generate your access token")
    print()

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        return None

    # Get credentials
    email = input("Enter your email: ")
    password = getpass("Enter your password: ")

    print("\nSigning in...")
    access_token, refresh_token = sign_in(email, password)

    if access_token:
        print("✅ Sign in successful!")
        global SUPABASE_REFRESH_TOKEN
        print("Testing token...")

        if test_token(access_token):
            SUPABASE_REFRESH_TOKEN = refresh_token or SUPABASE_REFRESH_TOKEN
            update_env_file(access_token, refresh_token)
            return access_token, refresh_token
        print("❌ Token validation failed.")
        return None
    print("❌ Sign in failed. Please check your credentials.")
    return None


def ensure_valid_token():
    """있는 .env 파일에 대해 검증 후 필요 시 refresh → signin 순으로 토큰 갱신.
    Returns:
        Tuple[str, str] | (None, None)
    """
    global SUPABASE_JWT_SECRET, SUPABASE_REFRESH_TOKEN

    # 1) 현재 토큰이 있고 유효하면 그대로 사용
    if SUPABASE_JWT_SECRET:
        print("Testing existing token...")
        if test_token(SUPABASE_JWT_SECRET):
            print("✅ Existing token is valid!")
            return SUPABASE_JWT_SECRET, SUPABASE_REFRESH_TOKEN
        else:
            print("❌ Existing token is invalid or expired.")

    # 2) refresh 토큰이 있으면 먼저 재발급 시도
    if SUPABASE_REFRESH_TOKEN:
        print("🔄 Trying to refresh access token with refresh token...")
        new_access, new_refresh = refresh_access_token(SUPABASE_REFRESH_TOKEN)
        if new_access:
            SUPABASE_JWT_SECRET = new_access
            if new_refresh:
                SUPABASE_REFRESH_TOKEN = new_refresh
            os.environ["SUPABASE_JWT_SECRET"] = new_access
            os.environ["SUPABASE_REFRESH_TOKEN"] = SUPABASE_REFRESH_TOKEN
            update_env_file(SUPABASE_JWT_SECRET, SUPABASE_REFRESH_TOKEN)
            if test_token(SUPABASE_JWT_SECRET):
                print("✅ Token refreshed and valid!")
                return SUPABASE_JWT_SECRET, SUPABASE_REFRESH_TOKEN
            else:
                print("❌ Refreshed token still invalid; will sign in.")
        else:
            print("❌ Refresh attempt failed; will sign in.")

    # 3) 둘 다 실패하면 사용자 인증 진행
    print("\n⚠️  No valid token found. Please authenticate.")
    creds = get_access_token()
    if creds:
        access_token, refresh_token = creds
        SUPABASE_JWT_SECRET = access_token
        SUPABASE_REFRESH_TOKEN = refresh_token
        os.environ["SUPABASE_JWT_SECRET"] = SUPABASE_JWT_SECRET
        os.environ["SUPABASE_REFRESH_TOKEN"] = SUPABASE_REFRESH_TOKEN
        return SUPABASE_JWT_SECRET, SUPABASE_REFRESH_TOKEN

    return None, None


# JWT exp 디코딩
def _decode_jwt_exp(token: str) -> Optional[datetime]:
    """JWT의 exp(만료 시각, epoch seconds)를 UTC datetime으로 디코딩. 실패 시 None."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        # Base64url padding 보정
        def _b64pad(s: str) -> bytes:
            pad = "=" * (-len(s) % 4)
            return (s + pad).encode()
        payload = json.loads(base64.urlsafe_b64decode(_b64pad(parts[1])).decode())
        exp = payload.get("exp")
        if isinstance(exp, (int, float)):
            return datetime.utcfromtimestamp(exp)
        return None
    except Exception:
        return None


# 리프레시 토큰으로 액세스 토큰 재발급 (비동기)
async def async_refresh_access_token(refresh_token: str):
    """리프레시 토큰으로 액세스 토큰을 재발급(비동기). API는 /auth/refresh 를 가정.
    백엔드는 refresh_token을 **쿼리 파라미터**로 받는다.
    """
    try:
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.post(
                f"{API_BASE_URL}/auth/refresh",
                params={"refresh_token": refresh_token},
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("access_token"), data.get("refresh_token") or refresh_token
            else:
                try:
                    msg = resp.json()
                except Exception:
                    msg = {"detail": resp.text}
                print(f"❌ Refresh failed ({resp.status_code}): {msg}")
                return None, None
    except Exception as e:
        print(f"❌ Refresh error: {e!s}")
        return None, None

# 리프레시 토큰으로 액세스 토큰 재발급 (동기)
def refresh_access_token(refresh_token: str):
    """리프레시 토큰으로 액세스 토큰을 재발급(동기). 백그라운드 스레드에서 사용.
    백엔드는 refresh_token을 **쿼리 파라미터**로 받는다.
    """
    try:
        resp = requests.post(
            f"{API_BASE_URL}/auth/refresh",
            params={"refresh_token": refresh_token},
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("access_token"), data.get("refresh_token") or refresh_token
        else:
            try:
                msg = resp.json()
            except Exception:
                msg = {"detail": resp.text}
            print(f"❌ Refresh failed ({resp.status_code}): {msg}")
            return None, None
    except Exception as e:
        print(f"❌ Refresh error: {e!s}")
        return None, None


# HTTP client
class LangConnectClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def update_token(self, token: str):
        """Update the authorization token."""
        self.token = token
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        else:
            self.headers.pop("Authorization", None)

    async def request(self, method: str, endpoint: str, **kwargs):
        global SUPABASE_JWT_SECRET, SUPABASE_REFRESH_TOKEN

        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}{endpoint}"
            try:
                response = await client.request(
                    method, url, headers=self.headers, timeout=60.0, **kwargs
                )
                response.raise_for_status()
                return response.json() if response.status_code != 204 else {"status": "success"}
            except httpx.HTTPStatusError as e:
                # 401이면 리프레시 후 1회 재시도
                if e.response is not None and e.response.status_code == 401 and SUPABASE_REFRESH_TOKEN:
                    print("🔄 401 Unauthorized detected — attempting token refresh...")
                    new_access, new_refresh = await async_refresh_access_token(SUPABASE_REFRESH_TOKEN)
                    if new_access:
                        # 전역 갱신
                        SUPABASE_JWT_SECRET = new_access
                        if new_refresh:
                            SUPABASE_REFRESH_TOKEN = new_refresh
                        # 헤더 갱신 및 .env 저장
                        self.update_token(new_access)
                        update_env_file(new_access, SUPABASE_REFRESH_TOKEN)
                        # 재시도
                        retry_resp = await client.request(
                            method, url, headers=self.headers, timeout=60.0, **kwargs
                        )
                        retry_resp.raise_for_status()
                        return retry_resp.json() if retry_resp.status_code != 204 else {"status": "success"}
                    # 리프레시 실패 시 원래 에러 재발생
                raise


def start_token_refresher():
    """백그라운드에서 JWT 만료 5분 전에 자동 갱신 시도."""
    def _loop():
        global SUPABASE_JWT_SECRET, SUPABASE_REFRESH_TOKEN
        while True:
            # 다음 갱신까지 대기 시간 계산
            exp_dt = _decode_jwt_exp(SUPABASE_JWT_SECRET) if SUPABASE_JWT_SECRET else None
            # 기본 대기: 55분 (토큰 1시간 기준), exp 파싱 성공 시 exp-5분
            if exp_dt:
                # exp는 UTC 기준
                now = datetime.utcnow()
                wait_s = max(60, int((exp_dt - now).total_seconds()) - 300)
            else:
                wait_s = 55 * 60
            time.sleep(wait_s)

            if not SUPABASE_REFRESH_TOKEN:
                print("⚠️ No refresh token available; cannot auto-refresh. Will retry later.")
                # 10분 뒤 재시도
                time.sleep(600)
                continue

            print("🔄 Attempting scheduled token refresh...")
            new_access, new_refresh = refresh_access_token(SUPABASE_REFRESH_TOKEN)
            if new_access:
                SUPABASE_JWT_SECRET = new_access
                if new_refresh:
                    SUPABASE_REFRESH_TOKEN = new_refresh
                client.update_token(SUPABASE_JWT_SECRET)
                update_env_file(SUPABASE_JWT_SECRET, SUPABASE_REFRESH_TOKEN)
                print("✅ Access token refreshed successfully.")
            else:
                print("❌ Scheduled token refresh failed; will retry in 5 minutes.")
                time.sleep(300)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()



# Initialize client (will be updated with valid token on startup)
client = LangConnectClient(API_BASE_URL, "")

# --- Client registry to decouple tools from direct global 'client' usage ---
_CLIENT: Optional[LangConnectClient] = None

def set_client(c: LangConnectClient) -> None:
    """Register the HTTP client for use inside tool functions (so they can be moved to other modules)."""
    global _CLIENT
    _CLIENT = c

def get_client() -> LangConnectClient:
    """Retrieve the registered HTTP client. Raises if not set."""
    if _CLIENT is None:
        raise RuntimeError("HTTP client not initialized yet")
    return _CLIENT
# --------------------------------------------------------------------------


# Create FastMCP server
mcp = FastMCP(
    name="mcp-rag",
    instructions="""이 서버는 대교씨엔에스 회사의 문서를 조회할 수 있는 기능을 제공합니다.

📌 Start here: how_to_use()
→ 해당 도구를 실행한 후 결과 내용을 토대로 문서 조회를 수행하세요.
""",
)


# Basic dynamic resource returning a string
# @mcp.resource(
#     "resource://how-to-use-rag-mcp",
#     name="RAG Playbook",
#     description="반드시 따라야 하는 단계별 사용 가이드입니다. 먼저 확인해서 가이드라인을 따라 도구를 사용하세요."
# )
# def get_instructions() -> str:
#     """반드시 따라야 하는 단계별 사용 가이드입니다. 가이드라인과 도구 사용 방법을 확인하세요."""
#     return """
# # How to Use

# Follow these steps:
# 1. Call `list_collections` to discover and choose the correct **Collection ID** for the user's request.
# 2. Call `multi_query(question)` to generate at least 3 sub‑questions related to the original request.
# 3. For each sub‑question, call `search_documents(collection_id, query, limit=5, search_type="hybrid")` and collect the most relevant passages.
# 4. Synthesize the final answer using the searched documents.

# **Output rules**
# - Answer in the same language as the user's request.
# - Append the sources you referenced at the very end.
# """
@mcp.tool(
        name="get-info",
        description="""이 MCP는 대교씨엔에스 회사의 문서 조회를 수행하는 도구입니다."""
)
async def get_info() -> str:
    return """
이 MCP는 대교씨엔에스 회사의 문서 조회를 수행하는 도구입니다.
how_to_use()를 호출하여 각 도구의 사용 방법과 가이드라인을 확인한 후 다음 도구를 진행하세요.
"""


@mcp.tool(
        name="how-to-use",
        description="""단계별 사용 가이드입니다. 반드시 how-to-use를 먼저 호출해서 각 도구의 사용 방법과 가이드라인을 확인하세요.""",
)
async def how_to_use() -> str:
    return """
# How to Use

Follow these steps:
1.	list_collections를 호출하여 사용자의 요청에 적합한 collection_id를 확인합니다.
2.	multi_query(question)을 호출하여 원래 질문과 관련된 하위 질문 3개 이상을 생성합니다.
3.	각 하위 질문에 대해 search_documents(collection_id, query, limit=5, search_type="hybrid")를 실행하여 가장 관련성이 높은 문서를 검색합니다.
4.	검색된 문서들이 사용자의 질문에 직접적이고 구체적으로 답하고 있는지 평가하세요. 사용자의 질문을 충족시키는 내용이 아니라면 적절하지 않다고 판단하세요. 
5.	적절하지 않은 경우, 다른 collection_id에 대해 search_documents(collection_id, query, limit=5, search_type="hybrid")를 다시 수행하세요.
6.	질문의 의도와 직접적으로 관련된 문서를 찾을 때까지 반복하세요.
7.	적절한 문서들이 수집되었다면 이를 기반으로 답변하세요.

**Output rules**
- Append the sources you referenced at the very end.
"""


@mcp.prompt("rag-prompt")
async def rag_prompt(query: str) -> list[dict]:
    """Provides a prompt for summarizing the provided text."""
    return [
        {
            "role": "system",
            "content": """You are a question-answer assistant based on given document.
You must use search tool to answer the question.

#Search Configuration:
- Target Collection: (user's request)
- Search Type: hybrid(preferred)
- Search Limit: 5(default)

#Search Guidelines:
Follow the guidelines step-by-step to find the answer.
1. Use `list_collections` to list up collections and find right **Collection ID** for user's request.
2. Use `multi_query` to generate at least 3 sub-questions which are related to original user's request.
3. Search all queries generated from previous step(`multi_query`) and find useful documents from collection.
4. Use searched documents to answer the question.

---

## Format:
(answer to the question)

**Source**
- [1] (Source and page numbers)
- [2] (Source and page numbers)
- ...

---

[Note]
- Answer in same language as user's request
- Append sources that you've referenced at the very end of your answer.
- If you can't find your answer from <search_results>, just say you can't find any relevant source to answer the question without any narrative sentences.
""",
        },
        {"role": "user", "content": f"User's request:\n\n{query}"},
    ]


@mcp.tool
async def search_documents(
    collection_id: str,
    query: str,
    limit: int = 5,
    search_type: str = "semantic",
    filter_json: Optional[str] = None,
) -> str:
    """
    Search documents in a collection using semantic, keyword, or hybrid search.

    Example:
    ```
    {
      "collection_id": "<UUID from list_collections()>",
      "query": "연차 사용 기준",
      "limit": 5,
      "search_type": "semantic"
    }
    ```
    """

    search_data = {"query": query, "limit": limit, "search_type": search_type}

    if filter_json:
        try:
            search_data["filter"] = json.loads(filter_json)
        except json.JSONDecodeError:
            return "Error: Invalid JSON in filter parameter"

    results = await get_client().request(
        "POST", f"/collections/{collection_id}/documents/search", json=search_data
    )

    if not results:
        return "No results found."

    output = f"## Search Results ({search_type})\n\n"
    for i, result in enumerate(results, 1):
        output += f"### Result {i} (Score: {result.get('score', 0):.4f})\n"
        output += f"{result.get('page_content', '')}\n"
        output += f"Document ID: {result.get('id', 'Unknown')}\n\n"

    return output


@mcp.tool(
        name="list_collections",
        description="""확인할 수 있는 모든 문서 collections를 조회합니다. """,
)
async def list_collections() -> str:
    """List all available document collections."""
    collections = await get_client().request("GET", "/collections")

    if not collections:
        return "No collections found."

    output = "## Collections\n\n"
    for coll in collections:
        meta = coll.get('metadata')
        if not isinstance(meta, dict):
            meta = {}
        description = meta.get('description') or 'Unknown'

        output += (
            f"- **{coll.get('name', 'Unnamed')}** (ID: {coll.get('uuid', 'Unknown')}) (Description: {description})\n"
        )

    return output


@mcp.tool(
        name="get_collection",
        description="""특정 collection의 상세 정보를 확인합니다."""
)
async def get_collection(collection_id: str) -> str:
    """Get details of a specific collection."""
    collection = await get_client().request("GET", f"/collections/{collection_id}")
    return f"**{collection.get('name', 'Unnamed')}**\nID: {collection.get('uuid', 'Unknown')}"


@mcp.tool(
        name="create_collection",
        description="""문서 DB에 collection을 생성합니다.""",
        enabled=False,
)
async def create_collection(name: str, metadata_json: Optional[str] = None) -> str:
    """Create a new collection."""
    data = {"name": name}

    if metadata_json:
        try:
            data["metadata"] = json.loads(metadata_json)
        except json.JSONDecodeError:
            return "Error: Invalid JSON in metadata"

    result = await get_client().request("POST", "/collections", json=data)
    return f"Collection '{result.get('name')}' created with ID: {result.get('uuid')}"


@mcp.tool
async def delete_collection(collection_id: str) -> str:
    """Delete a collection and all its documents."""
    await get_client().request("DELETE", f"/collections/{collection_id}")
    return f"Collection {collection_id} deleted successfully!"


@mcp.tool
async def list_documents(collection_id: str, limit: int = 20) -> str:
    """List documents in a collection."""
    docs = await get_client().request(
        "GET", f"/collections/{collection_id}/documents", params={"limit": limit}
    )

    if not docs:
        return "No documents found."

    output = f"## Documents ({len(docs)} items)\n\n"
    for i, doc in enumerate(docs, 1):
        content_preview = doc.get("page_content", "")[:200]
        if len(doc.get("page_content", "")) > 200:
            content_preview += "..."
        output += f"{i}. {content_preview}\n   ID: {doc.get('id', 'Unknown')}\n\n"

    return output


@mcp.tool
async def add_documents(collection_id: str, text: str) -> str:
    """Add a text document to a collection."""
    metadata = {"source": "mcp-input", "created_at": datetime.now().isoformat()}

    files = [("files", ("document.txt", text.encode("utf-8"), "text/plain"))]
    data = {"metadatas_json": json.dumps([metadata])}

    # Remove Content-Type for multipart
    headers = get_client().headers.copy()
    headers.pop("Content-Type", None)

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{get_client().base_url}/collections/{collection_id}/documents",
            headers=headers,
            files=files,
            data=data,
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()

    if result.get("success"):
        return f"Document added successfully! Created {len(result.get('added_chunk_ids', []))} chunks."
    return f"Failed to add document: {result.get('message', 'Unknown error')}"


@mcp.tool
async def delete_document(collection_id: str, document_id: str) -> str:
    """Delete a document from a collection."""
    await get_client().request(
        "DELETE", f"/collections/{collection_id}/documents/{document_id}"
    )
    return f"Document {document_id} deleted successfully!"


@mcp.tool
async def multi_query(question: str) -> str:
    """Generate multiple queries (3-5) for better vector search results from a single user question."""
    if not OPENAI_API_KEY:
        return json.dumps({"error": "OpenAI API key not configured"})

    try:
        # Initialize LLM
        llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)

        # Create prompt template
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 3 to 5 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Do not number them.
Original question: {question}""",
        )

        # Create parser
        output_parser = LineListOutputParser()

        # Create chain
        chain = query_prompt | llm | output_parser

        # Generate queries
        queries = await chain.ainvoke({"question": question})

        # Return as JSON array
        return json.dumps(queries, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Failed to generate queries: {e!s}"})


@mcp.tool
async def get_health_status() -> str:
    """Check API health status."""
    result = await get_client().request("GET", "/health")
    return f"Status: {result.get('status', 'Unknown')}\nAPI: {API_BASE_URL}\nAuth: {'✓' if SUPABASE_JWT_SECRET else '✗'}"


if __name__ == "__main__":
    print("🚀 MCP Server Using Streamable HTTP")
    print("=" * 50)

    # Ensure we have a valid token before starting
    valid_token, refresh_token = ensure_valid_token()

    if not valid_token:
        print("\n❌ Unable to obtain valid authentication token.")
        print("Please check your credentials and try again.")
        sys.exit(1)

    # Update the client with the valid token
    client.update_token(valid_token)
    set_client(client)

    # 갱신 스케줄러 시작
    start_token_refresher()
    print("🕒 Token auto-refresh background task started.")

    print(f"\n✅ Starting MCP server on http://127.0.0.1:{HTTP_PORT}/mcp")
    print(
        "This server is for MCP clients only and cannot be accessed directly in a browser."
    )
    print("\nℹ️  Access tokens expire in ~1 hour, but auto-refresh is enabled.")
    print("If refresh fails repeatedly, you'll be prompted to sign in again on next restart.")

    try:
        # NOTE: For Streamable HTTP, ensure uvicorn/fastapi are installed.
        # Port must be an integer; add explicit path for the HTTP endpoint.
        mcp.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=HTTP_PORT,          # ensure int, not string
            path="/mcp",
            log_level="debug",
        )
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)
