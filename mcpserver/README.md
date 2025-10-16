# FastMCP 기반 HTTP 서버 & stdio

## HTTP 서버

향후 사용이 편리하도록 http 기반으로 서버 설정<Br>
지금은 streamable-http 로 했는데, 버전 올리면 http로 바뀌는 것 같음. 나중에 변경 필요<Br>

### 설명

- 메인 코드 : mcp_http_server.py
- 프록시 서버 : proxy_server.py (클로드 데스크탑용)
  - 클로드 데스크탑의 경우 로컬에 별도 프록시 서버가 필요함(https://wikidocs.net/288326)
  - Claude Desktop의 설정 파일인 claude_desktop_config.json에는 반드시 command 필드가 포함되어야 하기 때문

#### 원격 MCP 적용
```python
# 클로드 데스크탑 이외에는 간단하게 아래 코드로 추가
# cursor 기준
# mcp.json 수정
{
  "mcpServers": {
    "mcp-rag": {
      "id": "mcp-rag",
      "name": "mcp-rag",
      "url": "http://remote-mcpserver-ip:4200/mcp", # 내 서버 IP
      "key": null,
      "enabled": true
    }
  }
}
```

#### 클로드 데스크탑 기준
```python
# 경로는 수정 필요
{
  "mcpServers": {
    "mcp-rag": {
      "command": "/Users/dkcns/.local/bin/uv",
      "args": [
          "--directory",
          "/Volumes/seuk_ssd/PycharmProjects/langconnect-client",
	  "run",
          "mcpserver/proxy_server.py"
      ]
    }
  }
}
```

### 실행 방법

- langconnect-client 또는 langconnect가 켜져 있는 상황에서, SUPABASE 계정이 준비된 상황
- `./run_mcp_http.sh` 실행


## Stdio 방식
기본 방식

### 설명
- 메인 코드 : mcp_stdio_server.py
  - 얘는 그때그때 필요할 때 마다 메인 코드를 run 하는 방식이라, mcp.json이 uv run으로 구성됨
- 서브 코드 : create_mcp_stdio_json.py
  - 이걸 실행해서 SUPABASE DB 안에 생성한 ID, PW를 적으면 1시간짜리 유효기간의 json 파일이 나옴
  - `mcp_stdio_config.json` 을 MCP에 적용하면 됨

### 실행 방법

- langconnect-client 또는 langconnect가 켜져 있는 상황에서, SUPABASE 계정이 준비된 상황
- `create_mcp_stdio_json.py` 실행해서 json 생성 후 MCP에 적용
- `./run_mcp_stdio.sh` 실행