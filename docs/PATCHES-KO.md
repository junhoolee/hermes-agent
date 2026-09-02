# 패치 배경 설명 (rebase/upstream-main-20260902 브랜치)

> 각 패치가 **왜 필요한지**의 기록. 업스트림 PR #80469이 업데이트되면 이 문서로 "아직 필요한 패치인가"를 판단하세요.
> 모든 패치는 TDD로 작성됨 — 각 커밋의 회귀 테스트는 패치 전 코드에서 실제로 실패합니다(커밋 메시지에 실패 출력 요약 있음). 적대적 교차 리뷰 2회 통과.
> 아래 커밋 sha 는 2026-09-02~03 리베이스(upstream/main 9/1 위로 재적용) 이후의 값입니다. 리베이스 전 `pr80469-patches` 시절의 sha 는 각 항목에 `(구 sha)` 로 병기합니다.

## 브랜치 구성 (2026-09-03 기준, 리베이스 후)

| 항목 | 값 |
|---|---|
| 기반 | upstream/main `ff7745fb0a` (2026-09-01) |
| PR #80469 | head `9326a55742` (base upstream/main `5538bd1f93`, 2026-08-15) 를 새 기반 위로 재적용 → `996192cbff`. 아직 OPEN, 미머지 |
| 자체 패치 | 22개 커밋 (`996192cbff..HEAD`) = 코드 커밋 18개 + 문서 커밋 4개. 번호 항목 1~17 중 6번은 문서, 15번은 폐기(아래), 나머지 15개 항목 유지 |
| 이전 구성 | `pr80469-patches` = upstream/main `acfd376d66`(7/29) + PR 원본 커밋 `330a533191` + 자체 패치 19개(`761517cfea` 까지). 리베이스 기록은 `hermes-rebase-notes/JOURNAL.md` |

PR 커밋 재적용(`996192cbff`) 시 PR head 대비 유일한 추가 변경: `scripts/check_claude_boundary.py` 허용목록에 `agent/anthropic_credentials.py` 추가.
업스트림 `7cbffdd125` 가 `agent/anthropic_adapter.py` 의 레거시 자격증명 코드를 그 파일로 분리했는데 PR 의 경계 검사 스크립트가 옛 파일만 알고 있어
`tests/scripts/test_claude_boundary.py` 가 깨졌기 때문. 규칙 자체는 건드리지 않았고 같은 `TODO(legacy-retirement)` 태그를 붙여 두었으므로 PR 이 업데이트되면 그쪽 허용목록과 대조.

## 배경: 업스트림 PR #80469이 미완성인 지점

PR #80469은 Claude **구독**(Pro/Max/Team)을 공식 Agent SDK로 연결하는 provider를 추가합니다. 설계 품질은 높지만,
"claude-code를 **폴백 슬롯**에 놓고 무인 게이트웨이로 돌리는" 구성은 작성자가 검증하지 않은 경로라 구멍이 있었습니다.
공통 원인 하나가 여러 곳에서 발현됩니다: **구독 런타임은 `api_key: ""`가 계약**(SDK가 유저의 claude 로그인을 스스로 해석)인데,
기존 코드 곳곳이 "api_key 없음 = provider 미설정"으로 가정합니다.

2차 패치 묶음(8월 하순, 7~12번)의 공통 주제는 다릅니다: codex 런타임에는 있는 **운영 안전장치**(heartbeat, 워치독,
/steer 전달, ack 자동 이어가기, 반복 상한, 실패 시 폴백)가 claude_agent_sdk 경로에는 없어서, 무인 운영 시 턴이 조용히
멈추거나 사용자 개입이 전달되지 않는 문제를 메운 것입니다.

3차(16~17번, 9/2)는 14번이 만든 폴백 루프의 빈 구멍 — SDK 가 예외 대신 `is_error` ResultMessage 로 알리는 세션 한도 —을 메웁니다.

---

## 1차 패치 (2026-08-06 ~ 08-09)

### 1. `0f73f9c379` (구 `60010cf`) — 브리지 툴 승인 컨텍스트 상실 (보안, 가장 중요)

- **증상**: 무인 게이트웨이에서 위험 명령(rm 계열 등)이 승인 절차 없이 **자동 승인**됨. 경고 로그 한 줄만 남음.
- **원인**: 브리지 툴 핸들러가 SDK 소유 이벤트루프 스레드에서 `propagate_context_to_thread`를 호출 — 이 함수의 계약은
  "부모(턴) 스레드에서 호출"인데, 루프 스레드엔 게이트웨이 ContextVar가 없어 승인 게이트가 fail-open 분기로 빠짐.
- **수정**: 턴 스레드에서(`_ensure_session`, 매 턴) 컨텍스트 스냅샷을 떠 agent에 저장 → 핸들러가 매 호출 `ctx.copy().run()`으로 사용 (병렬 툴콜 안전).
- **주의**: 기존 테스트는 `asyncio.run`(메인 스레드)이라 이 버그를 못 봅니다. 신규 테스트는 별도 스레드 루프에서 핸들러를 구동합니다.

### 2. `32fd174383` (구 `3594cfe`) — CLI 시작 타임아웃 설정화

- 60초 하드코딩 → `claude_subscription.start_timeout` config 키. 느린 호스트에서 세션 시작이 60초를 넘으면 턴이 죽습니다.

### 3. `187f3a74f6` + `8fc9303436` (구 `d0f30fa` + `737df72`) — 폴백 슬롯에서 claude-code 동작 (치명)

- **증상**: `fallback_providers`에 claude-code를 넣으면 폴백 발동 시 **매 턴 하드 실패**. 체인에 다음 프로바이더가 있으면
  claude를 건너뛰어버려 Claude가 한 턴도 서빙하지 못함.
- **원인 2겹**: ① `try_activate_fallback`의 api_mode 추론 사다리에 `claude_agent_sdk` 분기가 없고 entry의 `api_mode` 키도 무시
  ② 상태를 고쳐도 SDK 런타임 디스패치는 retry 루프 **앞**에 한 번만 실행되므로, 미드턴 폴백은 영영 SDK 경로에 못 들어감.
- **수정**: entry api_mode 존중(+오타는 경고 후 추론으로 강하), 그리고
  폴백 활성화가 claude_agent_sdk에 착지하면 현재 턴을 `_handoff_turn_to_claude_agent_sdk()`로 즉시 핸드오프 (콜사이트 10곳 전부).
- **리베이스로 일부 흡수됨(2026-09-02)**: 업스트림 #79787 이 entry `api_mode` 존중을 `resolve_provider_client()` 호출 앞의 pre-resolve 단계(`fb_api_mode_explicit`)로 독립 구현했고 폴백 사다리도 재작성했다.
  우리 패치의 "사다리 전체를 `determine_api_mode()` 로 교체" 부분은 업스트림 사다리와 중복이라 리베이스 때 버렸다. **남은 델타**만 유지: (a) 재판정 사다리 첫 rung —
  `determine_api_mode(...) == "claude_agent_sdk"` (base_url 이 불투명한 `claude-sdk://subscription` 이라 URL 사다리로는 못 잡음), (b) `claude_agent_sdk` 클라이언트 분기(client=None),
  (c) claude_agent_sdk 를 떠날 때 `_release_claude_agent_sdk_session()` 호출, (d) anthropic_messages 분기의 `normalize_provider()` 슬러그 정규화, (e) 선언된 `api_mode` 를 `TRANSPORT_TO_API_MODE` 로 검증(오타 → 경고 후 추론),
  (f) `conversation_loop.py` 의 미드턴 핸드오프 클로저 + 콜사이트 10곳. 업스트림에는 `claude_agent_sdk` 문자열이 없으므로 (a)(b)(c)(f) 는 PR 머지 전까지 전부 필요.
- **의도적 설계**: primary 복구 시 SDK 세션을 해제하지 않음(warm-keep) — 장기 rate-limit 중 매 턴 CLI 재기동을 피하기 위함. 에이전트 캐시 evict 시 해제됨.

### 4. `f4ade5dca2` + `67472f74d8` (구 `5dd7bbf` + `5f2b6b7`) — 압축·백그라운드의 keyless 오거부 3곳

- **증상**: `/compress` → "No provider configured -- cannot compress." / 자동(hygiene) 압축은 **로그 한 줄 없이 침묵 스킵** /
  백그라운드 태스크 거부. 무인 운영에서 압축이 죽으면 세션이 무한 성장해 결국 stall — 이 provider를 쓰는 이유 자체가 무력화됨.
- **원인**: 세 게이트 모두 `runtime_kwargs["api_key"]` 부재 = 미설정으로 판정.
- **수정**: 공유 `_runtime_is_keyless()`(api_mode==claude_agent_sdk) 예외 + 진짜 무설정은 계속 거부 + 침묵 스킵을 warning 로그로.
- 참고: 요약 생성 자체는 aux 자동감지가 메인 프로바이더(claude-code)로 원샷 SDK 클라이언트를 만들어 수행 — 이 경로는 업스트림에 이미 있고 정상.

### 5. `246a7e7c73` (구 `49d5fc6`) — 컨텍스트 토큰 과대추정 교정

- **증상**: 실제 ~2.6만 토큰 세션이 ~107만으로 표시, 자동 압축이 오탐 발화.
- **원인**: SDK `ResultMessage.usage`는 그 턴의 **모든 내부 API 호출 합산**(캐시 읽기 중복 포함)인데 이를 라이브 컨텍스트로 오용.
- **수정**: `AssistantMessage.usage`(호출당 개별)를 추적해 **마지막 호출**의 input+cache_read+cache_write를 컨텍스트 크기로 보고.
  빌링 누계는 기존 cumulative 그대로(미접촉). per-call usage가 없는 구형 CLI는 cumulative 폴백.

### 6. `ada1868cbb` + `f0d215df2d` (구 `c20ee89` + `c3c8bf3`) — 문서 (INSTALL-claude-subscription-KO.md, 본 문서)

---

## 2차 패치 (2026-08-19 ~ 09-01)

### 7. `73121ccacd` (구 `cd99d07`) — Slack: 강조 변환 전에 bare URL을 `<url>`로 감싸기 (Claude 무관)

- **증상**: Slack 어댑터가 마크다운 강조(`_`, `*`)를 변환할 때 URL 안의 `_`까지 건드려 링크가 깨짐.
- **수정**: 강조 변환 전에 bare URL을 Slack 링크 문법 `<url>`로 먼저 감싸 보호. 유일하게 claude_agent_sdk와 무관한 패치.
- **리베이스 판정(2026-09-03)**: 폐기 후보였으나 **유지**. upstream/main `ff7745fb0a` 의 `plugins/platforms/slack/adapter.py` 위에서 이 패치 없이 회귀 테스트 4건을 돌리면
  3건 실패(`test_bare_bold_url_does_not_swallow_closing_asterisk`, `test_bare_url_wrapped_in_angle_brackets`, `test_bare_url_with_balanced_parens_not_truncated`) — 업스트림의 8/15 이후 Slack 변경(unfurl 제어, ts claim 등)은 이 증상을 다루지 않음.
- 업스트림 `plugins/platforms/slack/adapter.py` 변경 시 여전히 필요한지 재확인.
- **리베이스 후속(2026-09-03)**: 업스트림 `tests/conformance/test_vector_generator.py::test_committed_vectors_match_regeneration` 이 커밋된 Slack 벡터와 렌더러 출력의 일치를 요구하므로
  `tests/conformance/vectors/slack.json` 을 `scripts/generate_conformance_vectors.py` 로 재생성해 이 커밋에 합쳤다(fixup). 바뀐 벡터는 `bare-url` 하나:
  `Visit https://example.com/path?q=1&amp;r=2 today.` → `Visit <https://example.com/path?q=1&r=2> today.` — 이 패치의 의도 그대로.
  이 벡터는 gateway-gateway 커넥터가 parity 기준으로 소비하므로 커넥터 쪽 Slack 렌더러를 쓰는 환경이면 같은 변경이 필요함(우리 환경은 네이티브 어댑터 사용).

### 8. `6c7ddd3789` (구 `167e9eb`) — auxiliary: 비동기 vision 경로에서 구독 shim 변환 누락

- **증상**: claude-code provider에서 vision 호출이 **간헐적으로** "Connection error"로 실패 (동기 경로는 정상, 비동기 경로만 실패).
- **원인**: `_to_async_client`에 구독 shim 분기가 없어 generic 변환기가 shim을 `AsyncOpenAI(base_url='claude-sdk://subscription')`로 감쌈 → httpx UnsupportedProtocol.
- **수정**: Codex/Anthropic/Bedrock 어댑터처럼 shim을 `AsyncClaudeAuxiliaryClient`로 라우팅해 원샷 SDK 경로 유지.

### 9. `74d9ac0a1a` (구 `c567853`) — 런타임: SDK 이벤트로 last-activity heartbeat

- **증상**: 범용 staleness 워치독(kanban worker reclaim, delegate heartbeat 진단)이 길게 도는 정상 Claude 턴을 stall로 오판.
- **원인**: `_dispatch()`가 `agent._touch_activity`를 한 번도 호출하지 않음.
- **수정**: 디스패치되는 모든 SDK 메시지에서 touch, 초당 1회로 throttle (StreamEvent는 content-block delta마다 옴). `_fire()` 경유라 메서드 없는 경량 stand-in도 안전.

### 10. `7139a3d800` (구 `abde6bd`) — 런타임: 침묵 SDK 턴용 stall 워치독

- **증상**: CLI 서브프로세스가 중간에 멎으면(wedged pipe 등) 유일한 가드가 1800초 턴 데드라인이라 30분간 "생각 중"으로 보임. codex에는 TTFB 워치독이 있는데 Claude 턴엔 없었음.
- **수정**: `run_turn(stall_timeout, stall_exempt)` — `stall_timeout`(기본 300초, `claude_subscription.stall_timeout`, 0이면 비활성) 동안 SDK 메시지가 없으면 데드라인과 구분되는 TimeoutError를 올려 기존 `should_retire=True` 처리로 라우팅.
- **오탐 방지**: 브리지 툴 호출 중엔 정상적으로 SDK 메시지가 없으므로, `run_bridged_tool`이 agent에 inflight 카운터(lock 보호, 병렬 read-tool 배치 대응)를 유지하고 `stall_exempt=lambda: inflight > 0`으로 연결.

### 11. `1bf2858b68` (구 `63c04cb`) — 런타임: `/steer`를 브리지 툴 결과와 턴 결과로 전달

- **증상**: 턴 진행 중 사용자가 보낸 `/steer`가 claude_agent_sdk 경로에서는 모델에 도달하지 않음.
- **수정**: `claude_tool_bridge._append_pending_steer`가 다음 브리지 툴 결과에 steer를 붙여 즉시 전달. 마지막 툴 호출 이후에 도착했거나 툴을 안 쓴 턴이면 `_attach_leftover_steer`가 `result["pending_steer"]`로 넘겨 chat_completions 경로와 동일한 핸드오프 계약을 따름.

### 12. `684f5fe327` (구 `0a4c115`) — 런타임: 중간 ack 자동 이어가기 (codex 루프와 동등)

- **증상**: 모델이 "확인해 볼게요..." 같은 의도 선언만 하고 행동 없이 턴을 끝내면 그대로 종료. codex 경로는 `codex_ack_continuations < 2`로 자동 이어가는데 Claude 경로엔 없었음.
- **수정**: `run_claude_agent_sdk_turn`이 최종 텍스트가 중간 ack로 보이면 같은 SDK 세션에 최대 2회 재질의(중간 텍스트는 interim assistant 메시지로 방출, 동일 continuation 프롬프트를 user 턴으로 추가). continuation 실패는 1차 결과로 강하(턴 실패 아님).
- **게이트**: `intent_ack_continuation_mode`(opt-in) + `claude_subscription.ack_continuation` kill switch(기본 on). `api_calls`는 실제 run_turn 횟수(1~3) 반영, 시도별 회계는 `_record_claude_attempt` 헬퍼로 공유.

### 13. `fa56a125ae` (구 `8b45fc4`) — 런타임: SDK 내부 반복 횟수 노출 및 상한

- **내용**: `ClaudeAgentSession`이 SDK 내부 반복(assistant 턴) 수를 `iteration_count`로 추적하고 `max_internal_iterations` 상한 도달 시 `iteration_cap_exceeded` 플래그. 기존 `tool_iterations`(개별 툴콜 수)와 별개 지표. 무한 루프 방어 + 관측성.

### 14. `1893aced03` (구 `1f6b616`) — 폴백: 실패한 claude_agent_sdk 턴을 폴백 체인으로 넘기기

- **증상**: SDK 턴 실패(preflight/빌링 거부, 세션 생성 오류)가 `fallback_providers`가 있어도 사용자에게 바로 에러로 반환. 3번(codex→claude 핸드오프)의 역방향이 없었음.
- **수정**: 디스패치를 루프로 — 체인의 다른 claude_agent_sdk 엔트리는 SDK 블록 재시도(다른 모델 핀 가능), codex_app_server는 직접 디스패치, HTTP 런타임은 표준 retry 루프로 낙하(`_sanitize_api_messages`가 dangling tool_calls 정리). 사용자 인터럽트는 재라우팅하지 않음.
- **알려진 한계(F2 검토, 2026-09-02)**: 이 커밋은 `conversation_loop.py` 만 바꿨고 `claude_runtime.py` 의 `run_turn` 예외/타임아웃 경로(stall·턴 데드라인)는 `failed=True` 를 세우지 않는다 —
  즉 stall/타임아웃 턴은 지금도 폴백을 타지 않고 partial 로 사용자에게 돌아간다. 부분 작업(툴콜 결과)이 이미 persist 된 뒤일 수 있어 폴백 provider 의 재실행 정책과 함께 설계 결정이 필요한 별도 후속 과제.
- **리베이스 후속(2026-09-03)**: 업스트림 #84733 소스 가드(`tests/agent/test_prompt_cache_ttl_propagation.py::TestFailoverRestartsPreflight`)가 `run_conversation` 안의 모든 `_try_activate_fallback` 참조를
  `if agent._try_activate_fallback(...):` + `continue`/`break` 형태로 강제한다. 이 커밋의 SDK 디스패치 루프는 `if not agent._try_activate_fallback(): return sdk_result` 로 쓰여 있어 가드에 걸렸으므로
  `if agent._try_activate_fallback(): …; continue` / `return sdk_result` 로 뒤집어 합쳤다(fixup, 동작 동일 — `continue` 는 `while agent.api_mode == "claude_agent_sdk"` 조건을 재평가할 뿐이고, SDK 가 아닌 폴백으로 빠지면 바깥 루프 첫머리의 pre-API preflight 가 그대로 실행된다).

### 15. ~~`da4df15`~~ — 모델 카탈로그: anthropic/claude-fable-5.1 추가 — **폐기 (upstream `9f069a1175` 에 포함, 기반 `ff7745fb0a` 에 이미 존재)**

- 업스트림 `9f069a11` 을 cherry-pick 한 것이었음. 리베이스 후 `hermes_cli/models.py` 와 `website/static/api/model-catalog.json` 의 fable-5.1 항목이 전부 upstream/main 에 있어
  커밋 잔여분이 `model-catalog.json` 의 `updated_at` 타임스탬프 한 줄뿐이었으므로 2026-09-03 리베이스에서 drop.

---

## 3차 패치 (2026-09-02, F1/F2 태스크)

### 16. `a1dee03cd8` (구 `eadb6b1a47`) — 폴백: is_error ResultMessage(세션 한도 등)를 실패로 취급

- **증상**: Claude Code CLI는 한도 초과 시 예외를 던지지 않고 `is_error=True` + `result=<에러 문구>`인 `ResultMessage`를 정상 스트림으로 보낸다. `run_claude_agent_sdk_turn`은 이를 `completed=False` partial 응답으로 조용히 사용자에게 반환할 뿐 `"failed"`를 세우지 않아, 14번 패치가 만든 폴백 디스패치 루프(`sdk_result.get("failed")`가 참일 때만 `_try_activate_fallback()` 호출)가 전혀 발동하지 않았다 — 14번의 빈 구멍.
- **수정**: `run_turn`이 예외 없이 끝났고 `projector.is_error`가 참이며 `iteration_cap_exceeded`가 거짓이고 사용자 인터럽트가 아닌 경우를 새 헬퍼 `_sdk_turn_reported_failure`로 판정해 턴 실패로 취급. `_record_claude_attempt`에 `persist` 플래그를 추가해 이 경우 에러 문구를 담은 assistant 메시지가 `messages`/세션DB에 남지 않게 하고(폴백 provider가 같은 trailing user 메시지부터 다시 처리하므로 role alternation 유지), `_retire_session`으로 SDK 세션을 폐기하고, `failed=True`인 `_failure_result`를 반환한다. iteration cap으로 SDK가 `is_error`를 세우는 경우(사용자가 아닌 Hermes가 요청한 인터럽트)는 기존대로 실패가 아니며, run_turn 예외 경로(기존 turn_error/stale-session recovery)는 손대지 않았다.
- **폐기 기준**: 업스트림 `run_claude_agent_sdk_turn`이 is_error 결과에 `failed=True`를 반환하면 폐기.
- **검증**: `tests/run_agent/test_provider_fallback.py::test_a_session_limit_is_error_result_hands_the_turn_to_openai_codex`(`00a4099322`, 구 `327454614b`)가 실제 `run_claude_agent_sdk_turn`을 통과시켜 세션 한도 ResultMessage → `openai-codex/gpt-5.6-sol` 핸드오프를 E2E로 재현한다.

### 17. `531c49db2e` (구 `f98eddeb00`) — 폴백 후속: cap·continuation 경로의 is_error 처리 보정

- **증상**: 16번과 같은 클래스의 두 갭. (1) iteration cap 인터럽트로 SDK가 `is_error`를 세우면 `_record_claude_attempt`가 `projector.error`를 `turn_error`로 복사해, `completed` 식의 cap 예외 주석("is_error is ignored whenever the cap fired")과 달리 `completed=False`/`partial=True`/`error=<abort 문구>`로 반환됐다(`failed=True` 오발동은 없었음). (2) ack-continuation의 2번째 `run_turn`이 세션 한도 등으로 `is_error` ResultMessage를 돌려주면 그 에러 문구가 assistant 행으로 `messages`/세션DB에 남고 `completed=False`로 떨어지며 폴백도 발동하지 않았다.
- **수정**: (1) `turn_error` 복사에도 `iteration_cap_exceeded` 예외를 적용. (2) continuation 결과에 `_sdk_turn_reported_failure`를 적용해 참이면 예외 arm과 동일 계약으로 처리 — continuation 프롬프트 pop, usage/compaction 회계만 기록(`persist=False`), 세션 retire, attempt 1의 완료 결과를 그대로 보고. 한도가 지속되면 다음 턴의 첫 attempt가 16번 경로로 폴백 체인에 넘긴다.
- **폐기 기준**: 16번과 함께 폐기(업스트림이 is_error를 실패로 다루고 cap/continuation을 구분하면).

---

## 업스트림 추적

- 원본 PR: https://github.com/NousResearch/hermes-agent/pull/80469
  - 최초 기반 커밋: `330a533191` (7월 말 base) / 현재 추적 중인 PR head: `9326a55742` (2026-08-15 base로 리베이스됨) → 우리 브랜치에서는 `996192cbff` 로 재적용됨
  - 우리 브랜치 기반: upstream/main `ff7745fb0a` (2026-09-01). upstream/main 의 `*.py` 에 `claude_agent_sdk` 문자열 0건 — PR 미머지 상태이므로 Claude 관련 패치는 모두 필요.
- PR이 업데이트/머지되면: 위 증상별 회귀 테스트를 새 코드에서 돌려보세요 — 통과하면 해당 패치는 폐기 가능.
  테스트 위치: `tests/run_agent/test_provider_fallback.py`(3, 14, 16번), `tests/agent/test_claude_tool_bridge.py`(1, 10, 11번),
  `tests/gateway/test_compress_command.py`·`test_session_hygiene.py`·`test_background_task_runtime_gate.py`(4번),
  `tests/agent/test_claude_runtime.py`(2, 5, 9~13, 16, 17번), `tests/agent/test_claude_auxiliary.py`(8번), `tests/gateway/test_slack.py`(7번).
- 폐기 판단 기준(2026-09-03 리베이스에서 적용한 방식): "패치 없이 그 패치의 회귀 테스트가 통과하는가". upstream/main 을 `/tmp` 에 별도 worktree 로 체크아웃하고
  우리 테스트 파일을 그 위에 복사해 돌리면 패치 코드 없이 테스트만 새 업스트림에 대고 실행할 수 있다(7번 판정에 사용).
- 리베이스 절차 (PR head 가 다시 바뀔 때): `git rebase --onto <새 PR head 재적용 커밋> 996192cbff` 로 자체 패치를 옮기거나, 새 upstream/main 위로 통째로 옮길 때는
  `GIT_EDITOR=true git rebase --onto upstream/main <이전 기반>` 두 단계(PR 커밋 → 자체 패치)로. 폐기할 커밋은
  `GIT_SEQUENCE_EDITOR="sed -i '' '/^pick <sha7>/s/^pick/drop/'" GIT_EDITOR=true git rebase -i <기반>` 으로 non-interactive drop.
  충돌 규모 사전 측정은 `git merge-tree --write-tree --merge-base=<commit>^ <새 기반> <commit>` 로 패치별 시뮬레이션 가능.
- 환경 의존 테스트(패치 무관): `tests/agent/test_claude_tool_bridge.py::test_read_only_annotation_matches_hermes_read_only_set` 은 `mcp==1.26.0` 이 설치된 venv 에서는
  순수 PR head 에서도 실패했다(`ToolAnnotations` 가 `read_only_hint` snake_case 별칭을 노출하지 않음). upstream/main 9/1 의 의존성으로 `pip install -e .` 를 다시 돌려 `mcp==2.0.0` 이 된 뒤에는 통과(2026-09-03 재확인 3/3).
  라이브 venv 교체 시 `mcp` 버전이 올라가는지 확인할 것.
- `uv.lock` 은 PR 커밋 해결 시 마커만 제거하고 양쪽 항목을 유지한 상태 — `uv lock` 재생성이 아직 안 됨(작업 머신에 uv 미설치). 라이브 적용 전 재생성 권장.
