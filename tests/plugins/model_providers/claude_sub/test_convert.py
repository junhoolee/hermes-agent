"""Message -> prompt conversion and usage accounting for claude-sub.

The SDK session speaks one text prompt per turn — or, when the newest user
message carries an image, a ``StreamPrompt`` of stream-json frames (v0.1-G).
These tests pin the framing (``<operating_instructions>`` /
``<prior_conversation>``), that tool messages never leak into the replayed
history, truncation of long histories, the image-block extraction/frame
contract, and the usage-accounting contract from a raw SDK usage dict
(last-call, not turn-cumulative).
"""

from __future__ import annotations

import asyncio


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


class TestImageBlocksFromContent:
    def test_non_list_content_returns_empty(self, load_plugin_module):
        convert = load_plugin_module("convert")
        assert convert.image_blocks_from_content("just text") == []
        assert convert.image_blocks_from_content(None) == []

    def test_data_url_base64_becomes_image_block(self, load_plugin_module):
        convert = load_plugin_module("convert")
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        assert convert.image_blocks_from_content(content) == [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
            }
        ]

    def test_jpeg_mime_is_preserved(self, load_plugin_module):
        convert = load_plugin_module("convert")
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBBB"}}]
        blocks = convert.image_blocks_from_content(content)
        assert blocks[0]["source"]["media_type"] == "image/jpeg"

    def test_missing_mime_defaults_to_png(self, load_plugin_module):
        convert = load_plugin_module("convert")
        content = [{"type": "image_url", "image_url": {"url": "data:;base64,CCCC"}}]
        blocks = convert.image_blocks_from_content(content)
        assert blocks[0]["source"]["media_type"] == "image/png"

    def test_http_url_becomes_url_source(self, load_plugin_module):
        convert = load_plugin_module("convert")
        content = [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]
        assert convert.image_blocks_from_content(content) == [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}
        ]

    def test_non_base64_data_url_becomes_note_text(self, load_plugin_module):
        convert = load_plugin_module("convert")
        content = [{"type": "image_url", "image_url": {"url": "data:image/png,rawbytes"}}]
        assert convert.image_blocks_from_content(content) == [
            {"type": "text", "text": "[an image was attached but could not be encoded]"}
        ]

    def test_file_path_string_becomes_note_text(self, load_plugin_module):
        convert = load_plugin_module("convert")
        content = [{"type": "image_url", "image_url": "/tmp/local.png"}]
        assert convert.image_blocks_from_content(content) == [
            {"type": "text", "text": "[an image was attached but could not be encoded]"}
        ]

    def test_empty_url_becomes_note_text(self, load_plugin_module):
        convert = load_plugin_module("convert")
        content = [{"type": "image_url", "image_url": {"url": ""}}]
        assert convert.image_blocks_from_content(content) == [
            {"type": "text", "text": "[an image was attached but could not be encoded]"}
        ]

    def test_input_image_type_is_recognized(self, load_plugin_module):
        convert = load_plugin_module("convert")
        content = [{"type": "input_image", "image_url": {"url": "https://example.com/b.png"}}]
        blocks = convert.image_blocks_from_content(content)
        assert blocks[0]["type"] == "image"

    def test_non_image_parts_are_ignored(self, load_plugin_module):
        convert = load_plugin_module("convert")
        content = [{"type": "text", "text": "hello"}]
        assert convert.image_blocks_from_content(content) == []


def _collect_frames(stream_prompt):
    async def _collect():
        return [frame async for frame in stream_prompt]

    return asyncio.run(_collect())


class TestStreamPrompt:
    def test_text_and_image_count_properties(self, load_plugin_module):
        convert = load_plugin_module("convert")
        prompt = convert.StreamPrompt(
            [
                {"type": "text", "text": "hi"},
                {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
            ]
        )
        assert prompt.text == "hi"
        assert prompt.image_count == 1

    def test_repr_does_not_leak_base64_data(self, load_plugin_module):
        convert = load_plugin_module("convert")
        long_data = "A" * 5000
        prompt = convert.StreamPrompt(
            [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": long_data}}]
        )
        assert long_data not in repr(prompt)
        assert repr(prompt) == "StreamPrompt(text_chars=0, images=1)"

    def test_reiteration_yields_fresh_frame_not_mutated_by_prior_send(self, load_plugin_module):
        convert = load_plugin_module("convert")
        prompt = convert.StreamPrompt([{"type": "text", "text": "hi"}])
        frame1 = _collect_frames(prompt)[0]
        frame1["session_id"] = "x"  # simulate the SDK mutating the sent frame
        frame2 = _collect_frames(prompt)[0]
        assert "session_id" not in frame2
        assert frame1["message"]["content"] == frame2["message"]["content"]


class TestMakePrompt:
    def test_no_images_joins_text_and_notes(self, load_plugin_module):
        convert = load_plugin_module("convert")
        result = convert.make_prompt(
            "hello", [{"type": "text", "text": "[an image was attached but could not be encoded]"}]
        )
        assert result == "hello\n[an image was attached but could not be encoded]"

    def test_no_images_no_notes_returns_text_unchanged(self, load_plugin_module):
        convert = load_plugin_module("convert")
        assert convert.make_prompt("hello", []) == "hello"

    def test_with_image_returns_stream_prompt(self, load_plugin_module):
        convert = load_plugin_module("convert")
        image_block = {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
        result = convert.make_prompt("hello", [image_block])
        assert isinstance(result, convert.StreamPrompt)
        assert result.blocks == [{"type": "text", "text": "hello"}, image_block]

    def test_with_image_and_empty_text_omits_text_block(self, load_plugin_module):
        convert = load_plugin_module("convert")
        image_block = {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}
        result = convert.make_prompt("", [image_block])
        assert result.blocks == [image_block]


class TestBuildPromptWithImages:
    def test_last_user_message_image_becomes_stream_prompt(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "system", "content": "You are Hermes."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert isinstance(prompt, convert.StreamPrompt)

        frames = _collect_frames(prompt)
        assert len(frames) == 1
        frame = frames[0]
        assert frame["type"] == "user"
        assert frame["message"]["role"] == "user"
        assert "session_id" not in frame
        assert frame["parent_tool_use_id"] is None
        content = frame["message"]["content"]
        assert content[0]["type"] == "text"
        assert content[0]["text"].startswith("<operating_instructions>")
        assert content[1] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
        }

    def test_image_omitted_placeholder_dropped_when_image_actually_encoded(
        self, load_plugin_module
    ):
        convert = load_plugin_module("convert")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert isinstance(prompt, convert.StreamPrompt)
        text_block = prompt.blocks[0]
        assert text_block["type"] == "text"
        assert "look at this" in text_block["text"]
        assert "[image omitted]" not in text_block["text"]
        assert any(b["type"] == "image" for b in prompt.blocks)

    def test_jpeg_mime_preserved_in_build_prompt(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBBB"}}]}
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert isinstance(prompt, convert.StreamPrompt)
        image_blocks = [b for b in prompt.blocks if b["type"] == "image"]
        assert image_blocks[0]["source"]["media_type"] == "image/jpeg"

    def test_http_url_in_build_prompt(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert isinstance(prompt, convert.StreamPrompt)
        image_blocks = [b for b in prompt.blocks if b["type"] == "image"]
        assert image_blocks[0]["source"] == {"type": "url", "url": "https://example.com/a.png"}

    def test_non_base64_data_url_falls_back_to_str_prompt(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "data:image/png,rawbytes"}},
                ],
            }
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert isinstance(prompt, str)
        assert "[an image was attached but could not be encoded]" in prompt

    def test_file_path_string_falls_back_to_str_prompt(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": "/tmp/local.png"}]}]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert isinstance(prompt, str)
        assert "[an image was attached but could not be encoded]" in prompt

    def test_image_on_non_last_user_message_is_omitted_as_text(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "earlier"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "final question"},
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        assert isinstance(prompt, str)
        assert "[image omitted]" in prompt

    def test_retransmission_safety_across_two_iterations(self, load_plugin_module):
        convert = load_plugin_module("convert")
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
        ]
        prompt = convert.build_prompt(messages, bootstrap_max_chars=60000)
        frame1 = _collect_frames(prompt)[0]
        frame1["session_id"] = "x"
        frame2 = _collect_frames(prompt)[0]
        assert "session_id" not in frame2


class TestBuildFollowupPrompt:
    def test_no_images_returns_joined_text(self, load_plugin_module):
        convert = load_plugin_module("convert")
        tail = [
            {"role": "assistant", "content": "prior reply"},
            {"role": "user", "content": "next question"},
        ]
        assert convert.build_followup_prompt(tail) == "prior reply\n\nnext question"

    def test_image_in_tail_becomes_stream_prompt(self, load_plugin_module):
        convert = load_plugin_module("convert")
        tail = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "check this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ],
            }
        ]
        prompt = convert.build_followup_prompt(tail)
        assert isinstance(prompt, convert.StreamPrompt)
        assert prompt.blocks[0]["type"] == "text"
        assert prompt.blocks[1] == {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/a.png"},
        }

    def test_image_omitted_placeholder_dropped_when_image_actually_encoded(
        self, load_plugin_module
    ):
        convert = load_plugin_module("convert")
        tail = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "check this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                ],
            }
        ]
        prompt = convert.build_followup_prompt(tail)
        assert isinstance(prompt, convert.StreamPrompt)
        text_block = prompt.blocks[0]
        assert text_block["type"] == "text"
        assert "check this" in text_block["text"]
        assert "[image omitted]" not in text_block["text"]

    def test_images_collected_in_order_across_tail_messages(self, load_plugin_module):
        convert = load_plugin_module("convert")
        tail = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/1.png"}}]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/2.png"}}]},
        ]
        prompt = convert.build_followup_prompt(tail)
        image_urls = [b["source"]["url"] for b in prompt.blocks if b["type"] == "image"]
        assert image_urls == ["https://x/1.png", "https://x/2.png"]


class TestTextFromContentImageOmission:
    def test_text_from_content_still_omits_image_as_text_marker(self, load_plugin_module):
        convert = load_plugin_module("convert")
        content = [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        assert convert.text_from_content(content) == "look\n[image omitted]"


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
