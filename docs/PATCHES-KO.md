# 패치 배경 설명 (pr80469-patches 브랜치)

> 각 패치가 **왜 필요한지**의 기록. 업스트림 PR #80469이 업데이트되면 이 문서로 "아직 필요한 패치인가"를 판단하세요.
> 모든 패치는 TDD로 작성됨 — 각 커밋의 회귀 테스트는 패치 전 코드에서 실제로 실패합니다(커밋 메시지에 실패 출력 요약 있음). 적대적 교차 리뷰 2회 통과.

## 브랜치 구성 (2026-09-02 기준)

| 항목 | 값 |
|---|---|
| 기반 | upstream/main `acfd376d66` (2026-07-29) + PR #80469 원본 커밋 `330a533191` |
| 자체 패치 | 18개 (`330a533191..HEAD`), 문서 2개 포함 |
| 추적 중인 PR head | `9326a55742` (base upstream/main `5538bd1f93`, 2026-08-15) — 아직 OPEN, 미머지 |
| 리베이스 목표 | `git rebase --onto <PR head> 330a533191` 로 PR 원본 커밋을 새 head로 교체 |

## 배경: 업스트림 PR #80469이 미완성인 지점

PR #80469은 Claude **구독**(Pro/Max/Team)을 공식 Agent SDK로 연결하는 provider를 추가합니다. 설계 품질은 높지만,
"claude-code를 **폴백 슬롯**에 놓고 무인 게이트웨이로 돌리는" 구성은 작성자가 검증하지 않은 경로라 구멍이 있었습니다.
공통 원인 하나가 여러 곳에서 발현됩니다: **구독 런타임은 `api_key: ""`가 계약**(SDK가 유저의 claude 로그인을 스스로 해석)인데,
기존 코드 곳곳이 "api_key 없음 = provider 미설정"으로 가정합니다.

2차 패치 묶음(8월 하순, 7~12번)의 공통 주제는 다릅니다: codex 런타임에는 있는 **운영 안전장치**(heartbeat, 워치독,
/steer 전달, ack 자동 이어가기, 반복 상한, 실패 시 폴백)가 claude_agent_sdk 경로에는 없어서, 무인 운영 시 턴이 조용히
멈추거나 사용자 개입이 전달되지 않는 문제를 메운 것입니다.

---

## 1차 패치 (2026-08-06 ~ 08-09)

### 1. `60010cf` — 브리지 툴 승인 컨텍스트 상실 (보안, 가장 중요)

- **증상**: 무인 게이트웨이에서 위험 명령(rm 계열 등)이 승인 절차 없이 **자동 승인**됨. 경고 로그 한 줄만 남음.
- **원인**: 브리지 툴 핸들러가 SDK 소유 이벤트루프 스레드에서 `propagate_context_to_thread`를 호출 — 이 함수의 계약은
  "부모(턴) 스레드에서 호출"인데, 루프 스레드엔 게이트웨이 ContextVar가 없어 승인 게이트가 fail-open 분기로 빠짐.
- **수정**: 턴 스레드에서(`_ensure_session`, 매 턴) 컨텍스트 스냅샷을 떠 agent에 저장 → 핸들러가 매 호출 `ctx.copy().run()`으로 사용 (병렬 툴콜 안전).
- **주의**: 기존 테스트는 `asyncio.run`(메인 스레드)이라 이 버그를 못 봅니다. 신규 테스트는 별도 스레드 루프에서 핸들러를 구동합니다.

### 2. `3594cfe` — CLI 시작 타임아웃 설정화

- 60초 하드코딩 → `claude_subscription.start_timeout` config 키. 느린 호스트에서 세션 시작이 60초를 넘으면 턴이 죽습니다.

### 3. `d0f30fa` + `737df72` — 폴백 슬롯에서 claude-code 동작 (치명)

- **증상**: `fallback_providers`에 claude-code를 넣으면 폴백 발동 시 **매 턴 하드 실패**. 체인에 다음 프로바이더가 있으면
  claude를 건너뛰어버려 Claude가 한 턴도 서빙하지 못함.
- **원인 2겹**: ① `try_activate_fallback`의 api_mode 추론 사다리에 `claude_agent_sdk` 분기가 없고 entry의 `api_mode` 키도 무시
  ② 상태를 고쳐도 SDK 런타임 디스패치는 retry 루프 **앞**에 한 번만 실행되므로, 미드턴 폴백은 영영 SDK 경로에 못 들어감.
- **수정**: entry api_mode 존중(+오타는 경고 후 추론으로 강하), `determine_api_mode()` 재사용, 그리고
  폴백 활성화가 claude_agent_sdk에 착지하면 현재 턴을 `_handoff_turn_to_claude_agent_sdk()`로 즉시 핸드오프 (콜사이트 10곳 전부).
- **의도적 설계**: primary 복구 시 SDK 세션을 해제하지 않음(warm-keep) — 장기 rate-limit 중 매 턴 CLI 재기동을 피하기 위함. 에이전트 캐시 evict 시 해제됨.

### 4. `5dd7bbf` + `5f2b6b7` — 압축·백그라운드의 keyless 오거부 3곳

- **증상**: `/compress` → "No provider configured -- cannot compress." / 자동(hygiene) 압축은 **로그 한 줄 없이 침묵 스킵** /
  백그라운드 태스크 거부. 무인 운영에서 압축이 죽으면 세션이 무한 성장해 결국 stall — 이 provider를 쓰는 이유 자체가 무력화됨.
- **원인**: 세 게이트 모두 `runtime_kwargs["api_key"]` 부재 = 미설정으로 판정.
- **수정**: 공유 `_runtime_is_keyless()`(api_mode==claude_agent_sdk) 예외 + 진짜 무설정은 계속 거부 + 침묵 스킵을 warning 로그로.
- 참고: 요약 생성 자체는 aux 자동감지가 메인 프로바이더(claude-code)로 원샷 SDK 클라이언트를 만들어 수행 — 이 경로는 업스트림에 이미 있고 정상.

### 5. `49d5fc6` — 컨텍스트 토큰 과대추정 교정

- **증상**: 실제 ~2.6만 토큰 세션이 ~107만으로 표시, 자동 압축이 오탐 발화.
- **원인**: SDK `ResultMessage.usage`는 그 턴의 **모든 내부 API 호출 합산**(캐시 읽기 중복 포함)인데 이를 라이브 컨텍스트로 오용.
- **수정**: `AssistantMessage.usage`(호출당 개별)를 추적해 **마지막 호출**의 input+cache_read+cache_write를 컨텍스트 크기로 보고.
  빌링 누계는 기존 cumulative 그대로(미접촉). per-call usage가 없는 구형 CLI는 cumulative 폴백.

### 6. `c20ee89` + `c3c8bf3` — 문서 (INSTALL-claude-subscription-KO.md, 본 문서)

---

## 2차 패치 (2026-08-19 ~ 09-01)

### 7. `cd99d07` — Slack: 강조 변환 전에 bare URL을 `<url>`로 감싸기 (Claude 무관)

- **증상**: Slack 어댑터가 마크다운 강조(`_`, `*`)를 변환할 때 URL 안의 `_`까지 건드려 링크가 깨짐.
- **수정**: 강조 변환 전에 bare URL을 Slack 링크 문법 `<url>`로 먼저 감싸 보호. 유일하게 claude_agent_sdk와 무관한 패치.
- 업스트림 `plugins/platforms/slack/adapter.py` 변경 시 여전히 필요한지 재확인.

### 8. `167e9eb` — auxiliary: 비동기 vision 경로에서 구독 shim 변환 누락

- **증상**: claude-code provider에서 vision 호출이 **간헐적으로** "Connection error"로 실패 (동기 경로는 정상, 비동기 경로만 실패).
- **원인**: `_to_async_client`에 구독 shim 분기가 없어 generic 변환기가 shim을 `AsyncOpenAI(base_url='claude-sdk://subscription')`로 감쌈 → httpx UnsupportedProtocol.
- **수정**: Codex/Anthropic/Bedrock 어댑터처럼 shim을 `AsyncClaudeAuxiliaryClient`로 라우팅해 원샷 SDK 경로 유지.

### 9. `c567853` — 런타임: SDK 이벤트로 last-activity heartbeat

- **증상**: 범용 staleness 워치독(kanban worker reclaim, delegate heartbeat 진단)이 길게 도는 정상 Claude 턴을 stall로 오판.
- **원인**: `_dispatch()`가 `agent._touch_activity`를 한 번도 호출하지 않음.
- **수정**: 디스패치되는 모든 SDK 메시지에서 touch, 초당 1회로 throttle (StreamEvent는 content-block delta마다 옴). `_fire()` 경유라 메서드 없는 경량 stand-in도 안전.

### 10. `abde6bd` — 런타임: 침묵 SDK 턴용 stall 워치독

- **증상**: CLI 서브프로세스가 중간에 멎으면(wedged pipe 등) 유일한 가드가 1800초 턴 데드라인이라 30분간 "생각 중"으로 보임. codex에는 TTFB 워치독이 있는데 Claude 턴엔 없었음.
- **수정**: `run_turn(stall_timeout, stall_exempt)` — `stall_timeout`(기본 300초, `claude_subscription.stall_timeout`, 0이면 비활성) 동안 SDK 메시지가 없으면 데드라인과 구분되는 TimeoutError를 올려 기존 `should_retire=True` 처리로 라우팅.
- **오탐 방지**: 브리지 툴 호출 중엔 정상적으로 SDK 메시지가 없으므로, `run_bridged_tool`이 agent에 inflight 카운터(lock 보호, 병렬 read-tool 배치 대응)를 유지하고 `stall_exempt=lambda: inflight > 0`으로 연결.

### 11. `63c04cb` — 런타임: `/steer`를 브리지 툴 결과와 턴 결과로 전달

- **증상**: 턴 진행 중 사용자가 보낸 `/steer`가 claude_agent_sdk 경로에서는 모델에 도달하지 않음.
- **수정**: `claude_tool_bridge._append_pending_steer`가 다음 브리지 툴 결과에 steer를 붙여 즉시 전달. 마지막 툴 호출 이후에 도착했거나 툴을 안 쓴 턴이면 `_attach_leftover_steer`가 `result["pending_steer"]`로 넘겨 chat_completions 경로와 동일한 핸드오프 계약을 따름.

### 12. `0a4c115` — 런타임: 중간 ack 자동 이어가기 (codex 루프와 동등)

- **증상**: 모델이 "확인해 볼게요..." 같은 의도 선언만 하고 행동 없이 턴을 끝내면 그대로 종료. codex 경로는 `codex_ack_continuations < 2`로 자동 이어가는데 Claude 경로엔 없었음.
- **수정**: `run_claude_agent_sdk_turn`이 최종 텍스트가 중간 ack로 보이면 같은 SDK 세션에 최대 2회 재질의(중간 텍스트는 interim assistant 메시지로 방출, 동일 continuation 프롬프트를 user 턴으로 추가). continuation 실패는 1차 결과로 강하(턴 실패 아님).
- **게이트**: `intent_ack_continuation_mode`(opt-in) + `claude_subscription.ack_continuation` kill switch(기본 on). `api_calls`는 실제 run_turn 횟수(1~3) 반영, 시도별 회계는 `_record_claude_attempt` 헬퍼로 공유.

### 13. `8b45fc4` — 런타임: SDK 내부 반복 횟수 노출 및 상한

- **내용**: `ClaudeAgentSession`이 SDK 내부 반복(assistant 턴) 수를 `iteration_count`로 추적하고 `max_internal_iterations` 상한 도달 시 `iteration_cap_exceeded` 플래그. 기존 `tool_iterations`(개별 툴콜 수)와 별개 지표. 무한 루프 방어 + 관측성.

### 14. `1f6b616` — 폴백: 실패한 claude_agent_sdk 턴을 폴백 체인으로 넘기기

- **증상**: SDK 턴 실패(preflight/빌링 거부, 세션 생성 오류, stall/턴 타임아웃)가 `fallback_providers`가 있어도 사용자에게 바로 에러로 반환. 3번(codex→claude 핸드오프)의 역방향이 없었음.
- **수정**: 디스패치를 루프로 — 체인의 다른 claude_agent_sdk 엔트리는 SDK 블록 재시도(다른 모델 핀 가능), codex_app_server는 직접 디스패치, HTTP 런타임은 표준 retry 루프로 낙하(`_sanitize_api_messages`가 dangling tool_calls 정리). 사용자 인터럽트는 재라우팅하지 않음.

### 15. `da4df15` — 모델 카탈로그: anthropic/claude-fable-5.1 추가

- 업스트림 `9f069a11` cherry-pick. OpenRouter/Nous 큐레이션 목록에 5.1 추가, 매니페스트는 `scripts/build_model_catalog.py`로 로컬 재생성(업스트림 json hunk 미적용). 리베이스 시 업스트림에 이미 있으면 폐기.

### 16. `eadb6b1a47` — 폴백: is_error ResultMessage(세션 한도 등)를 실패로 취급

- **증상**: Claude Code CLI는 한도 초과 시 예외를 던지지 않고 `is_error=True` + `result=<에러 문구>`인 `ResultMessage`를 정상 스트림으로 보낸다. `run_claude_agent_sdk_turn`은 이를 `completed=False` partial 응답으로 조용히 사용자에게 반환할 뿐 `"failed"`를 세우지 않아, 14번 패치가 만든 폴백 디스패치 루프(`sdk_result.get("failed")`가 참일 때만 `_try_activate_fallback()` 호출)가 전혀 발동하지 않았다 — 14번의 빈 구멍.
- **수정**: `run_turn`이 예외 없이 끝났고 `projector.is_error`가 참이며 `iteration_cap_exceeded`가 거짓이고 사용자 인터럽트가 아닌 경우를 새 헬퍼 `_sdk_turn_reported_failure`로 판정해 턴 실패로 취급. `_record_claude_attempt`에 `persist` 플래그를 추가해 이 경우 에러 문구를 담은 assistant 메시지가 `messages`/세션DB에 남지 않게 하고(폴백 provider가 같은 trailing user 메시지부터 다시 처리하므로 role alternation 유지), `_retire_session`으로 SDK 세션을 폐기하고, `failed=True`인 `_failure_result`를 반환한다. iteration cap으로 SDK가 `is_error`를 세우는 경우(사용자가 아닌 Hermes가 요청한 인터럽트)는 기존대로 실패가 아니며, run_turn 예외 경로(기존 turn_error/stale-session recovery)는 손대지 않았다.
- **폐기 기준**: 업스트림 `run_claude_agent_sdk_turn`이 is_error 결과에 `failed=True`를 반환하면 폐기.
- **검증**: `tests/run_agent/test_provider_fallback.py::test_a_session_limit_is_error_result_hands_the_turn_to_openai_codex`(`327454614b`)가 실제 `run_claude_agent_sdk_turn`을 통과시켜 세션 한도 ResultMessage → `openai-codex/gpt-5.6-sol` 핸드오프를 E2E로 재현한다.

### 17. `f98eddeb00` — 폴백 후속: cap·continuation 경로의 is_error 처리 보정

- **증상**: 16번과 같은 클래스의 두 갭. (1) iteration cap 인터럽트로 SDK가 `is_error`를 세우면 `_record_claude_attempt`가 `projector.error`를 `turn_error`로 복사해, `completed` 식의 cap 예외 주석("is_error is ignored whenever the cap fired")과 달리 `completed=False`/`partial=True`/`error=<abort 문구>`로 반환됐다(`failed=True` 오발동은 없었음). (2) ack-continuation의 2번째 `run_turn`이 세션 한도 등으로 `is_error` ResultMessage를 돌려주면 그 에러 문구가 assistant 행으로 `messages`/세션DB에 남고 `completed=False`로 떨어지며 폴백도 발동하지 않았다.
- **수정**: (1) `turn_error` 복사에도 `iteration_cap_exceeded` 예외를 적용. (2) continuation 결과에 `_sdk_turn_reported_failure`를 적용해 참이면 예외 arm과 동일 계약으로 처리 — continuation 프롬프트 pop, usage/compaction 회계만 기록(`persist=False`), 세션 retire, attempt 1의 완료 결과를 그대로 보고. 한도가 지속되면 다음 턴의 첫 attempt가 16번 경로로 폴백 체인에 넘긴다.
- **폐기 기준**: 16번과 함께 폐기(업스트림이 is_error를 실패로 다루고 cap/continuation을 구분하면).

---

## 업스트림 추적

- 원본 PR: https://github.com/NousResearch/hermes-agent/pull/80469
  - 우리 기반 커밋: `330a533191` (7월 말 base) / 현재 PR head: `9326a55742` (2026-08-15 base로 리베이스됨, 코드 hunk 45개 변경)
- PR이 업데이트/머지되면: 위 증상별 회귀 테스트를 새 코드에서 돌려보세요 — 통과하면 해당 패치는 폐기 가능.
  테스트 위치: `tests/run_agent/test_provider_fallback.py`(3, 14, 16번), `tests/agent/test_claude_tool_bridge.py`(1, 10, 11번),
  `tests/gateway/test_compress_command.py`·`test_session_hygiene.py`·`test_background_task_runtime_gate.py`(4번),
  `tests/agent/test_claude_runtime.py`(2, 5, 9~13, 16, 17번), `tests/agent/test_claude_auxiliary.py`(8번), `tests/gateway/test_slack.py`(7번).
- 리베이스 절차: `git rebase --onto 9326a55742 330a533191 pr80469-patches` (PR 원본 커밋을 새 head로 교체). 충돌 규모 사전 측정은
  `git merge-tree --write-tree --merge-base=<commit>^ 9326a55742 <commit>` 로 패치별 시뮬레이션 가능.
