from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

anthropic = pytest.importorskip("anthropic")

from native_runner.review import __main__ as review_cli  # noqa: E402
from native_runner.review.commentary import (  # noqa: E402
    COMMENTARY_MODEL,
    CommentaryError,
    build_brief,
    render_commentary,
    request_commentary,
)
from native_runner.review.page import render_page  # noqa: E402
from native_runner.review.report import zone_label  # noqa: E402
from native_runner.tests.review_helpers import scripted_review  # noqa: E402


ANSWER = {
    "overview": "前一分钟你和对手互有往来，1:07 之后局势一路下滑。",
    "moments": [
        {"start": "1:07", "end": "1:17", "title": "右路防守没接住", "analysis": "这十秒局势值掉了 0.4。"},
        {"start": "2:26", "end": "2:26", "title": "反打的起点", "analysis": "万箭齐发之后局势开始回升。"},
        {"start": "接近终局", "end": "2:50", "title": "收尾", "analysis": "这里的时间写法无法定位。"},
    ],
    "habits": ["圣水攒到 9 以上才出牌"],
    "suggestions": ["对手下骑士后先在己方半场解掉", "两倍圣水阶段不要满圣水"],
    "caveats": "数据里没有单位交战过程，原因部分是推测。",
}


class ClaudeStub:
    """A local server that answers the Messages API the way the real one streams."""

    def __init__(self, *, stop_reason: str = "end_turn", text: str | None = None) -> None:
        self.requests: list[dict] = []
        body = json.dumps(ANSWER, ensure_ascii=False) if text is None else text
        half = len(body) // 2
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stub",
                    "type": "message",
                    "role": "assistant",
                    "model": COMMENTARY_MODEL,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 9000, "output_tokens": 1},
                },
            }
        ]
        if stop_reason != "refusal":
            events += [
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                *(
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": part}}
                    for part in (body[:half], body[half:])
                ),
                {"type": "content_block_stop", "index": 0},
            ]
        events += [
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 1200},
            },
            {"type": "message_stop"},
        ]
        payload = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events
        ).encode("utf-8")
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["content-length"])
                stub.requests.append(
                    {
                        "path": self.path,
                        "headers": {key.lower(): value for key, value in self.headers.items()},
                        "body": json.loads(self.rfile.read(length)),
                    }
                )
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "ClaudeStub":
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.server.server_close()

    def client(self) -> "anthropic.Anthropic":
        port = self.server.server_address[1]
        return anthropic.Anthropic(api_key="stub-key", base_url=f"http://127.0.0.1:{port}", max_retries=0)


@pytest.fixture(scope="module")
def review() -> dict:
    return scripted_review()


def test_brief_gives_claude_both_sides_in_the_reviewed_players_view(review: dict) -> None:
    brief = build_brief(review, 0)
    own, opponent = review["players"]

    assert brief["player"] == {"side": "team", "result": "win", "crowns": 1}
    assert (
        brief["fidelity"]["verdict"] == "faithful" and brief["fidelity"]["trusted_until"] == brief["match"]["duration"]
    )
    assert [action["time"] for action in brief["own_actions"]] == [action["time"] for action in own["actions"]]
    assert "top_cells" not in brief["own_actions"][0] and "cell" not in brief["own_actions"][0]
    picks = {
        action["model_pick"] if isinstance(action["model_pick"], str) else "other" for action in brief["own_actions"]
    }
    assert picks == {"same", "other"}

    # The opponent's cells are stored in the opponent's view; the brief turns them around.
    theirs = opponent["actions"][0]
    mirrored = [17 - theirs["cell"][0], 31 - theirs["cell"][1]]
    first = next(action for action in brief["opponent_actions"] if action["time"] == theirs["time"])
    assert first["zone"] == zone_label(mirrored) and first["zone"].startswith("对方")
    assert theirs["zone"].startswith("己方")
    scored_and_unscored = len(opponent["actions"]) + len(opponent["unscored_source_actions"])
    assert len(brief["opponent_actions"]) == scored_and_unscored

    # One row every five seconds keeps a three-minute match small enough to read.
    assert 30 <= len(brief["timeline"]) <= 40
    assert len(json.dumps(brief, ensure_ascii=False)) < 30_000
    assert brief["own_actions_not_scored"] and brief["full_elixir_periods"]
    assert all(period["start"] <= period["end"] for period in brief["full_elixir_periods"])
    assert [swing["time"] for swing in brief["largest_value_swings"]] == [swing["time"] for swing in own["swings"]]
    with pytest.raises(ValueError, match="no player 2"):
        build_brief(review, 2)


def test_brief_of_a_partial_reconstruction_stops_at_the_trusted_part() -> None:
    partial = scripted_review(partial=True)

    brief = build_brief(partial, 0)

    assert brief["fidelity"]["verdict"] == "partial"
    assert brief["fidelity"]["trusted_until"] == brief["timeline"][-1]["time"] < brief["match"]["duration"]
    assert brief["fidelity"]["failure_reason"] == "battle truncated before source timeline"


def test_request_goes_to_opus_with_fallbacks_and_a_schema_and_parses_the_stream(review: dict) -> None:
    brief = build_brief(review, 0)
    with ClaudeStub() as stub:
        commentary = request_commentary(brief, client=stub.client(), effort="medium")

    (request,) = stub.requests
    assert request["path"].startswith("/v1/messages")
    assert "server-side-fallback-2026-07-01" in request["headers"]["anthropic-beta"]
    body = request["body"]
    assert body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default" and body["stream"] is True
    assert body["output_config"]["effort"] == "medium"
    schema = body["output_config"]["format"]["schema"]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert set(schema["properties"]) == {"overview", "moments", "habits", "suggestions", "caveats"}
    assert "thinking" not in body and "temperature" not in body and "tool_choice" not in body
    assert "教练" in body["system"]
    (message,) = body["messages"]
    sent = json.loads(message["content"].split("\n\n", 1)[1])
    assert message["role"] == "user" and sent == json.loads(json.dumps(brief))

    assert commentary["model"] == COMMENTARY_MODEL and commentary["effort"] == "medium"
    assert commentary["usage"] == {"input_tokens": 9000, "output_tokens": 1200}
    assert commentary["overview"] == ANSWER["overview"]
    assert commentary["habits"] == ANSWER["habits"] and commentary["suggestions"] == ANSWER["suggestions"]
    first, second, unplaced = commentary["moments"]
    assert (first["time"], first["start_tick"], first["end_tick"]) == ("1:07–1:17", 67 * 20, 77 * 20)
    assert (second["time"], second["start_tick"], second["end_tick"]) == ("2:26", 146 * 20, 146 * 20)
    # A time Claude wrote loosely is kept as text but not linked to the timeline.
    assert (unplaced["time"], unplaced["start_tick"], unplaced["end_tick"]) == ("接近终局–2:50", None, None)

    text = render_commentary(commentary)
    assert text.startswith(ANSWER["overview"]) and "1:07–1:17  右路防守没接住" in text
    assert "  - 两倍圣水阶段不要满圣水" in text and text.endswith("局限：" + ANSWER["caveats"])


@pytest.mark.parametrize(
    ("stub", "message"),
    [
        (lambda: ClaudeStub(stop_reason="refusal"), "declined"),
        (lambda: ClaudeStub(stop_reason="max_tokens", text='{"overview": "写到一半'), "cut off"),
        (lambda: ClaudeStub(text='{"overview": "只有总评"}'), "did not match the commentary format"),
    ],
)
def test_refusals_and_truncated_answers_are_errors_not_commentary(review: dict, stub, message: str) -> None:
    with stub() as server, pytest.raises(CommentaryError, match=message):
        request_commentary(build_brief(review, 0), client=server.client())


def test_explain_command_stores_commentary_in_the_review_and_its_page(
    review: dict, monkeypatch, tmp_path: Path, capsys
) -> None:
    path = tmp_path / "match.review.json"
    path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")

    assert review_cli.main(["explain", str(path), "--show-request", "--player", "1"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["model"] == "claude-opus-5-5" and shown["output_config"]["format"]["schema"]["type"] == "object"
    assert "commentary" not in json.loads(path.read_text(encoding="utf-8"))["players"][1]

    with ClaudeStub() as stub:
        client = stub.client()
        monkeypatch.setattr(anthropic, "Anthropic", lambda: client)
        assert review_cli.main(["explain", str(path), "--player", "0"]) == 0
    output = capsys.readouterr().out
    assert "玩家 0 的讲解（claude-opus-5-5，输入 9000 / 输出 1200 tokens）" in output and ANSWER["overview"] in output
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["players"][0]["commentary"]["overview"] == ANSWER["overview"]
    assert "commentary" not in stored["players"][1] and len(stub.requests) == 1
    assert path.with_suffix(".html").read_text(encoding="utf-8") == render_page(stored)


@pytest.mark.skipif(
    (Path.home() / ".config" / "anthropic").exists(), reason="a stored Claude profile would supply credentials"
)
def test_explain_command_says_how_to_supply_a_key_when_none_is_set(
    review: dict, monkeypatch, tmp_path: Path, capsys
) -> None:
    path = tmp_path / "match.review.json"
    path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    # Nothing listens here, so the test cannot reach a real service whatever the SDK resolves.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")

    assert review_cli.main(["explain", str(path), "--player", "1"]) == 1

    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert json.loads(path.read_text(encoding="utf-8")) == review
