import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Run smoke checks against the Agentic RAG API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--token", default="dev-token")
    args = parser.parse_args()

    checks = [
        ("health", lambda: get_json(f"{args.base_url}/health")),
        ("readiness", lambda: get_json(f"{args.base_url}/health/ready")),
        ("tools", lambda: get_json(f"{args.base_url}/api/v1/tools", args.token)),
        (
            "chat",
            lambda: post_json(
                f"{args.base_url}/api/v1/chat",
                args.token,
                {"message": "请介绍一下这个项目的架构", "user_id": "smoke-user"},
            ),
        ),
    ]

    for name, check in checks:
        try:
            payload = check()
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            return 1
        print(f"[OK] {name}: {summarize(name, payload)}")

    return 0


def get_json(url: str, token: str | None = None) -> dict:
    request = urllib.request.Request(url, headers=headers(token))
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, token: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={**headers(token), "Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def headers(token: str | None = None) -> dict[str, str]:
    values = {"Accept": "application/json"}
    if token:
        values["Authorization"] = f"Bearer {token}"
    return values


def summarize(name: str, payload: dict) -> str:
    if name == "health":
        return payload.get("status", "unknown")
    if name == "readiness":
        rag = payload.get("rag", {})
        return f"{payload.get('status', 'unknown')}, rag={rag.get('backend', 'unknown')}"
    if name == "tools":
        return f"{len(payload.get('tools', []))} tools"
    if name == "chat":
        route = payload.get("route", {}).get("route")
        sources = len(payload.get("sources", []))
        return f"route={route}, sources={sources}"
    return "ok"


if __name__ == "__main__":
    sys.exit(main())
