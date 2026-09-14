# Hermes + Claude 구독 백본 설치 가이드 (pr80469-patches 브랜치)

> 이 브랜치 = 업스트림 PR #80469 (Claude subscription via official Agent SDK) + 실운영 검증에서 발견한 결함 수정 7커밋.
> 2026-08-09부터 무인 Slack 게이트웨이(Codex primary + Claude 폴백)에서 실운영 중인 구성입니다.

## 필요한 것

- Linux/macOS, **Python 3.11+**, git
- **Claude 구독 계정** (Pro/Max/Team — API 키 아님, 과금 $0)
- (선택) ChatGPT 구독 — Codex를 primary로 쓸 경우

## 설치

```bash
git clone -b pr80469-patches https://github.com/junhoolee/hermes-agent.git
cd hermes-agent
python3.11 -m venv .venv
.venv/bin/pip install -e '.[claude-code]'   # claude-agent-sdk >=0.2.128 자동 설치
```

## Claude 자격증명 (구독 로그인)

Hermes를 실행할 유저로 [Claude Code CLI](https://code.claude.com)를 설치하고 구독 계정으로 로그인:

```bash
claude auth login    # 브라우저에서 claude.ai 구독 계정 인증
claude auth status   # loggedIn: true, authMethod: claude.ai 확인
```

⚠️ **환경변수에 `ANTHROPIC_API_KEY`(또는 `ANTHROPIC_*` 일체)가 있으면 안 됩니다** — 실수 API 과금을 막는 정적 빌링 게이트가 Claude 턴을 전부 거부합니다. `.env`·셸 프로필에서 제거하세요.

## config.yaml 핵심 설정

`~/.hermes/config.yaml` (첫 실행 후 생성됨):

```yaml
# 필수 — 안 열면 claude-code가 조용히 구식 anthropic API 경로로 치환됩니다
claude_subscription:
  enabled: true
  start_timeout: 120        # CLI 세션 시작 대기(초). 느린 호스트는 120 권장 (기본 60)
```

**방법 A — Claude를 메인 모델로:**

```bash
hermes model    # claude-code / claude-sonnet-5 선택
```

**방법 B — Codex primary + Claude 폴백** (이 브랜치의 실운영 구성):

```yaml
model:
  default: openai-codex/gpt-5.6-sol
  provider: openai-codex
  base_url: https://chatgpt.com/backend-api/codex
fallback_providers:
  - provider: claude-code     # ⚠️ 셋 다 필수 — 하나라도 빠지면 entry 무시/오동작
    model: claude-sonnet-5
    api_mode: claude_agent_sdk
```

## 실행·확인

```bash
.venv/bin/hermes              # CLI 대화
.venv/bin/hermes gateway run  # Slack 등 게이트웨이 (SLACK_BOT_TOKEN/SLACK_APP_TOKEN은 ~/.hermes/.env)
```

정상 동작 확인 (대화 1턴 후):

```bash
.venv/bin/python -c "import sqlite3; con=sqlite3.connect('$HOME/.hermes/state.db'); \
  [print(r) for r in con.execute('SELECT model, billing_mode, actual_cost_usd FROM session_model_usage ORDER BY rowid DESC LIMIT 3')]"
# 기대값: ('claude-sonnet-5', 'subscription_included', 0.0)
```

## 이 브랜치가 업스트림 PR과 다른 점 (수정 7커밋)

| 커밋 | 수정 |
|---|---|
| `60010cf` | 브리지 툴 승인 컨텍스트 상실 → 위험 명령 자동승인(보안) 수정 |
| `3594cfe` | CLI 시작 타임아웃 config화 (`claude_subscription.start_timeout`) |
| `d0f30fa`+`737df72` | **폴백 슬롯에서 claude-code 동작하게** (원본은 폴백 진입 시 매 턴 하드 실패) |
| `5dd7bbf`+`5f2b6b7` | /compress·자동압축·백그라운드가 구독 런타임(api_key 없음)을 오거부하던 것 수정 |
| `49d5fc6` | 컨텍스트 토큰 ~N배 과대추정(자동압축 오탐 발화) 교정 |

## 함정 모음 (전부 실제로 밟은 것)

1. `claude_subscription.enabled` 안 열면 **조용히** anthropic API 경로로 감 (에러 없음)
2. 폴백 entry는 `provider`+`model`+`api_mode` 3키 전부 필요
3. `ANTHROPIC_*` 환경변수 → Claude 턴 전면 거부
4. 모델 id는 대시형만 (`claude-sonnet-5`, `claude-opus-4-8`)
5. config 수정 후엔 게이트웨이 **restart** (start 아님)
6. `hermes -z`(원샷 CLI)로는 폴백 테스트 불가 — 폴백은 게이트웨이 경유만 동작
7. 정지된 claude CLI 자식 프로세스 누적 가능 → `pgrep -f claude | wc -l` 주기 점검 권장
