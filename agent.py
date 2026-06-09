import json
import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

import anthropic
import streamlit as st
from tools import get_player_stats, predict_salary

def _get_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except:
        return os.environ.get("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=_get_api_key())

TOOLS = [
    {
        "name": "get_player_stats",
        "description": (
            "NBA 선수의 이름과 시즌 연도를 입력받아 nba_api에서 실시간 스탯을 조회합니다. "
            "반환값에는 경기당 PTS, TRB, AST 등 주요 스탯과 모델 입력용 파생 피처가 포함됩니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_name": {
                    "type": "string",
                    "description": "NBA 선수 이름 (영문). 예: 'LeBron James', 'Stephen Curry'"
                },
                "year": {
                    "type": "integer",
                    "description": "시즌 종료 연도. 예: 2026은 2025-26시즌. 기본값 2026."
                }
            },
            "required": ["player_name"]
        }
    },
    {
        "name": "predict_salary",
        "description": (
            "get_player_stats로 조회한 스탯 dict와 실제 연봉(USD)을 받아 "
            "ML 모델로 적정 연봉을 예측하고 먹튀/적정/저평가 판정을 반환합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stats": {
                    "type": "object",
                    "description": "get_player_stats의 반환값 전체"
                },
                "actual_salary": {
                    "type": "integer",
                    "description": "선수의 실제 연봉 (USD 정수). 모를 경우 생략 가능."
                }
            },
            "required": ["stats"]
        }
    }
]

SYSTEM_PROMPT = """당신은 NBA 선수 연봉 가치 분석 전문가입니다.
당신은 get_player_stats Tool을 통해 실시간으로 NBA 공식 API에 접근할 수 있습니다. 현재 시즌은 2025-26(year=2026)입니다.

규칙:
- 선수 이름이 언급되면 반드시 먼저 get_player_stats Tool을 호출하세요. 절대로 "조회할 수 없다"고 말하지 마세요.
- year는 항상 2026을 기본값으로 사용하세요.
- get_player_stats 결과를 받은 후 predict_salary Tool을 호출하세요.
- 도구 호출 전에 텍스트 답변을 먼저 출력하지 마세요.

분석 결과 형식:
- 선수 정보: 이름, 시즌, 포지션
- 주요 스탯: PTS/TRB/AST 등
- 예측 적정 연봉 vs 실제 연봉
- 가치 판정: 먹튀/적정/저평가
- 판정 근거: 어떤 스탯이 연봉을 정당화하는지, 혹은 못하는지 구체적으로 설명

항상 한국어로 답변하세요. 텍스트 중간에 상표 기호(TM), 저작권 기호(c) 같은 특수 기호를 단어 안에 삽입하지 마세요."""


def _execute_tool(name: str, inputs: dict):
    try:
        if name == "get_player_stats":
            return get_player_stats(
                player_name=inputs["player_name"],
                year=inputs.get("year", 2026)
            )
        elif name == "predict_salary":
            return predict_salary(
                stats=inputs["stats"],
                actual_salary=inputs.get("actual_salary")
            )
        else:
            return {"error": f"알 수 없는 Tool: {name}"}
    except Exception as e:
        return {"error": str(e)}


def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return "\n".join(b.text for b in response.content if hasattr(b, 'text'))

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            return "\n".join(b.text for b in response.content if hasattr(b, 'text'))

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            result = _execute_tool(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        messages.append({"role": "user", "content": tool_results})
