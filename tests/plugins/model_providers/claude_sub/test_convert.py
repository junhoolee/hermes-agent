"""Message -> prompt conversion and usage accounting for claude-sub.

The SDK session speaks one text prompt per turn. These tests pin the framing
(``<operating_instructions>`` / ``<prior_conversation>``), that tool messages
never leak into the replayed history, truncation of long histories, and the
usage-accounting contract from a raw SDK usage dict (last-call, not
turn-cumulative).
"""

from __future__ import annotations


class TestSplitMessages:
    def test_extracts_system_last_user_and_prior(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ]
        system_text, last_user_text, prior = convert.split_messages(messages)
        assert system_text == "You are Hermes."
        assert last_user_text == "second question"
        assert prior == [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]

    def test_includes_tool_messages_in_prior_except_last_user(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do a thing"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "tool result", "tool_call_id": "1"},
            {"role": "user", "content": "final question"},
        ]
        _system, last_user_text, prior = convert.split_messages(messages)
        assert last_user_text == "final question"
        assert prior == messages[1:4]
        assert any(m.get("role") == "tool" for m in prior)

    def test_first_user_text(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert convert.first_user_text(messages) == "first"

    def test_list_content_with_image_is_omitted_as_text_marker(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
        _system, last_user_text, _prior = convert.split_messages(messages)
        assert "look at this" in last_user_text
        assert "[image omitted]" in last_user_text


class TestBuildPrompt:
    def test_orders_operating_instructions_then_prior_then_last_user(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "final question"},
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        oi_index = prompt.index("<operating_instructions>")
        pc_index = prompt.index("<prior_conversation>")
        final_index = prompt.index("final question")
        assert oi_index < pc_index < final_index
        assert "You are Hermes." in prompt
        assert "earlier question" in prompt
        assert "earlier answer" in prompt

    def test_no_system_message_omits_operating_instructions(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [{"role": "user", "content": "hello"}]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert "<operating_instructions>" not in prompt
        assert prompt.endswith("hello")

    def test_no_prior_history_omits_prior_conversation_block(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "only question"},
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert "<prior_conversation>" not in prompt

    def test_tool_calls_and_results_included_in_replayed_history(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "read the secret file"},
            {
                "role": "assistant",
                "content": "calling a tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/secret.txt"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "TOOL_OUTPUT_TEXT", "tool_call_id": "call_1"},
            {"role": "user", "content": "final question"},
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert "TOOL_OUTPUT_TEXT" in prompt
        assert "[tool call id=call_1 name=read_file args=" in prompt
        assert '"path": "/tmp/secret.txt"' in prompt
        assert "Tool result (read_file): TOOL_OUTPUT_TEXT" in prompt

    def test_long_history_is_truncated_to_bootstrap_max_chars(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [{"role": "system", "content": "sys"}]
        for i in range(50):
            messages.append({"role": "user", "content": f"question {i}" * 20})
            messages.append({"role": "assistant", "content": f"answer {i}" * 20})
        messages.append({"role": "user", "content": "final question"})
        prompt = convert.build_prompt(messages, bootstrap_max_chars=500)
        prior_start = prompt.index("<prior_conversation>")
        prior_end = prompt.index("</prior_conversation>")
        # The truncation marker must appear, and the retained body must be
        # bounded near bootstrap_max_chars (plus the fixed framing text).
        assert "…" in prompt[prior_start:prior_end]
        assert (prior_end - prior_start) < 500 + 1000


class TestBootstrapToolHistory:
    def test_tool_call_args_truncated_to_500_chars(self, load_plugin_module):
        convert = load_plugin_module("convert")
        long_args = '{"path": "' + ("x" * 600) + '"}'
        messages = [
            {"role": "user", "content": "do a thing"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": long_args},
                    }
                ],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "call_1"},
            {"role": "user", "content": "final question"},
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        call_line = next(
            line for line in prompt.splitlines() if line.startswith("[tool call id=call_1")
        )
        args_text = call_line.split("args=", 1)[1].rstrip("]")
        assert len(args_text) == convert.BOOTSTRAP_TOOL_ARGS_MAX_CHARS
        assert args_text == long_args[: convert.BOOTSTRAP_TOOL_ARGS_MAX_CHARS]

    def test_no_text_assistant_message_renders_only_tool_call_lines(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "user", "content": "do a thing"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "call_1"},
            {"role": "user", "content": "final question"},
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert "Assistant:" not in prompt
        assert "[tool call id=call_1 name=read_file args={}]" in prompt

    def test_tool_result_truncated_to_2000_chars_with_marker(self, load_plugin_module):
        convert = load_plugin_module("convert")
        long_result = "y" * 3000
        messages = [
            {"role": "user", "content": "do a thing"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": long_result, "tool_call_id": "call_1"},
            {"role": "user", "content": "final question"},
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert "Tool result (read_file): " + ("y" * 2000) + " …[truncated]" in prompt
        assert "y" * 2001 not in prompt

    def test_tool_result_falls_back_to_tool_call_id_when_name_unresolved(
        self, load_plugin_module
    ):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "user", "content": "do a thing"},
            {"role": "tool", "content": "orphaned result", "tool_call_id": "call_missing"},
            {"role": "user", "content": "final question"},
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert "Tool result (call_missing): orphaned result" in prompt

    def test_prior_conversation_preamble_mentions_tool_history(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "final question"},
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        prior_start = prompt.index("<prior_conversation>")
        prior_end = prompt.index("</prior_conversation>")
        preamble = prompt[prior_start:prior_end]
        assert "already executed" in preamble
        assert "[tool call ...]" in preamble
        assert "Tool result (...)" in preamble

    def test_build_prompt_order_unaffected_by_tool_history(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "do a thing"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "tool output", "tool_call_id": "call_1"},
            {"role": "user", "content": "final question"},
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        oi_index = prompt.index("<operating_instructions>")
        pc_index = prompt.index("<prior_conversation>")
        final_index = prompt.rindex("final question")
        assert oi_index < pc_index < final_index
        assert prompt.endswith("final question")


class TestUsageFromAssistant:
    def test_combines_input_and_cache_tokens_into_prompt_tokens(self, load_plugin_module):
        convert = load_plugin_module("convert")
        usage = convert.usage_from_assistant(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 20,
            }
        )
        assert usage["prompt_tokens"] == 150
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 200
        assert usage["cached_tokens"] == 30

    def test_missing_usage_is_all_zero(self, load_plugin_module):
        convert = load_plugin_module("convert")
        usage = convert.usage_from_assistant(None)
        assert usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }

    def test_empty_dict_is_all_zero(self, load_plugin_module):
        convert = load_plugin_module("convert")
        assert convert.usage_from_assistant({}) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
