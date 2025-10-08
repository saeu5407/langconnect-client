from fastmcp import FastMCP

proxy = FastMCP.as_proxy(
    "http://127.0.0.1:4200/mcp", 
    name="H-RO Proxy",
)

if __name__ == "__main__":
    proxy.run()