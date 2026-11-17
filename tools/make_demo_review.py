"""Write a scripted example review and its report page.

The match and every number in it are invented.  Use it to look at the report
page, or to try ``python -m native_runner.review explain`` against the real
Claude API, before an engine-reconstructed review exists.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from native_runner.review.page import render_page
    from native_runner.tests.review_helpers import scripted_review

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("runs/demo-review"))
    args = parser.parse_args()
    review = scripted_review()
    review["replay_tag"] = "示例数据-不是真实对局"
    review["model"]["checkpoint_file"] = "无（数字由脚本编造）"
    review["match"]["game_mode"] = "虚构示例，仅用于展示页面"
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "example.review.json"
    path.write_text(json.dumps(review, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    path.with_suffix(".html").write_text(render_page(review), encoding="utf-8")
    print(f"Wrote {path} and {path.with_suffix('.html').name}; the match in them is invented.")


if __name__ == "__main__":
    main()
