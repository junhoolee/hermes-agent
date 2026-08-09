# 패치 배경 설명 (pr80469-patches 브랜치)

> 각 패치가 **왜 필요한지**의 기록. 업스트림 PR #80469이 업데이트되면 이 문서로 "아직 필요한 패치인가"를 판단하세요.
> 모든 패치는 TDD로 작성됨 — 각 커밋의 회귀 테스트는 패치 전 코드에서 실제로 실패합니다(커밋 메시지에 실패 출력 요약 있음). 적대적 교차 리뷰 2회 통과.

## 배경: 업스트림 PR #80469이 미완성인 지점

PR #80469은 Claude **구독**(Pro/Max/Team)을 공식 Agent SDK로 연결하는 provider를 추가합니다. 설계 품질은 높지만,
"claude-code를 **폴백 슬롯**에 놓고 무인 게이트웨이로 돌리는" 구성은 작성자가 검증하지 않은 경로라 구멍이 있었습니다.
공통 원인 하나가 여러 곳에서 발현됩니다: **구독 런타임은 `api_key: ""`가 계약**(SDK가 유저의 claude 로그인을 스스로 해석)인데,
기존 코드 곳곳이 "api_key 없음 = provider 미설정"으로 가정합니다.

---

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

### 6. `커밋 없음` — 문서 (INSTALL-claude-subscription-KO.md, 본 문서)

---

## 업스트림 추적

- 원본 PR: https://github.com/NousResearch/hermes-agent/pull/80469 (base sha `330a533191`)
- PR이 업데이트/머지되면: 위 증상별 회귀 테스트를 새 코드에서 돌려보세요 — 통과하면 해당 패치는 폐기 가능.
  테스트 위치: `tests/run_agent/test_provider_fallback.py`(3번), `tests/agent/test_claude_tool_bridge.py`(1번),
  `tests/gateway/test_compress_command.py`·`test_session_hygiene.py`·`test_background_task_runtime_gate.py`(4번),
  `tests/agent/test_claude_runtime.py`(2·5번).
