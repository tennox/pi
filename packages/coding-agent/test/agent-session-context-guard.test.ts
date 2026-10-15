/**
 * Tests for the pre-request context-exhaustion guard in AgentSession.
 *
 * The guard lives in `_enforceContextGuardBeforeRequest`, invoked from the
 * wrapped `prepareRequest` hook just after compaction has had its chance to
 * shrink context. When the projected outgoing context strictly exceeds the
 * model's raw context window, it throws the structured error
 *
 *   context exhausted: ~N tokens exceeds limit M — run cancelled (no automatic recovery)
 *
 * so the run fails fast with a real explanation instead of a provider 400 +
 * empty response (the original bug: 2,454 parallel tool calls → 8.95 MB of
 * results → 2.81M tokens against a 1M window → "No output"). The guard
 * deliberately fires only past the raw window (not past compaction's
 * `contextWindow - reserve` threshold) so the existing compaction flow keeps
 * its reserve accounting.
 */

import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import { createHarness, type Harness } from "./suite/harness.ts";

describe("AgentSession context-exhaustion guard", () => {
	const harnesses: Harness[] = [];
	afterEach(() => {
		while (harnesses.length) harnesses.pop()?.cleanup();
	});

	function seedUsage(harness: Harness, totalInputTokens: number): void {
		// Stamp the assistant message with usage that the projection will pick up,
		// then mirror it into the agent state. We don't go through the real LLM
		// stream here; we're testing the pre-request guard, not the API.
		const model = harness.session.model!;
		harness.sessionManager.appendMessage({
			role: "user",
			content: [{ type: "text", text: "seed user" }],
			timestamp: Date.now() - 2000,
		});
		const assistant = fauxAssistantMessage("seed assistant", { timestamp: Date.now() - 1000 });
		harness.sessionManager.appendMessage({
			...assistant,
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				...assistant.usage,
				input: totalInputTokens,
				totalTokens: totalInputTokens,
			},
		});
		harness.session.agent.state.messages = harness.sessionManager.buildSessionContext().messages;
	}

	it("aborts with the exact error message when projected context exceeds the raw window", async () => {
		// contextWindow=1000. Seed 1500 input tokens (exceeds raw window) with
		// compaction disabled so the guard path is taken instead of compaction.
		const harness = await createHarness({
			models: [{ id: "faux-1", contextWindow: 1000 }],
			tools: [],
			settings: {
				compaction: {
					enabled: false,
					reserveTokens: 100,
					keepRecentTokens: 20000,
				},
			},
		});
		harnesses.push(harness);
		seedUsage(harness, 1500);
		harness.setResponses([fauxAssistantMessage("should never reach the model")]);

		await harness.session.prompt("anything");
		const errorMessage = harness.session.agent.state.errorMessage;
		expect(errorMessage).toBeDefined();
		expect(errorMessage).toMatch(
			/context exhausted: ~\d+ tokens exceeds limit 1000 \u2014 run cancelled \(no automatic recovery\)/,
		);
		const lastAssistant = harness.session.messages.findLast((message) => message.role === "assistant");
		expect(lastAssistant?.role).toBe("assistant");
		if (lastAssistant?.role === "assistant") {
			expect(lastAssistant.stopReason).toBe("error");
			expect(lastAssistant.errorMessage).toMatch(
				/context exhausted: ~\d+ tokens exceeds limit 1000 \u2014 run cancelled \(no automatic recovery\)/,
			);
		}
	});

	it("also rejects when estimated tokens (no prior usage) cross the raw window", async () => {
		// No provider usage recorded (input=0) → projection falls back to a
		// per-message char/4 estimate. With enough seed text, the estimate crosses
		// the raw window and the guard fires.
		const harness = await createHarness({
			models: [{ id: "faux-1", contextWindow: 4000 }],
			tools: [],
			settings: {
				compaction: { enabled: false, reserveTokens: 100, keepRecentTokens: 20000 },
			},
		});
		harnesses.push(harness);
		const model = harness.session.model!;
		harness.sessionManager.appendMessage({
			role: "user",
			content: [{ type: "text", text: "x".repeat(12_000) }],
			timestamp: Date.now() - 2000,
		});
		harness.sessionManager.appendMessage({
			...fauxAssistantMessage("y".repeat(12_000), { timestamp: Date.now() - 1000 }),
			api: model.api,
			provider: model.provider,
			model: model.id,
		});
		harness.session.agent.state.messages = harness.sessionManager.buildSessionContext().messages;

		harness.setResponses([fauxAssistantMessage("never")]);
		await harness.session.prompt("more");
		expect(harness.session.agent.state.errorMessage).toMatch(
			/context exhausted: ~\d+ tokens exceeds limit 4000 \u2014 run cancelled \(no automatic recovery\)/,
		);
	});

	it("does not fire the guard for a normal small-turn path", async () => {
		// contextWindow=128_000; a single short prompt is far below it.
		const harness = await createHarness({
			models: [{ id: "faux-1", contextWindow: 128_000 }],
			tools: [],
		});
		harnesses.push(harness);
		harness.setResponses([fauxAssistantMessage("hello back")]);

		await harness.session.prompt("hi");

		expect(harness.session.agent.state.errorMessage).toBeUndefined();
		const assistants = harness.session.messages.filter((message) => message.role === "assistant");
		expect(assistants.length).toBeGreaterThan(0);
	});

	it("guards a contextWindow=0 model defensively (no crash, no abort)", async () => {
		// Models with no reported limit (contextWindow <= 0) must NOT trip the
		// guard — there's no ceiling to exceed. The provider is responsible for
		// any actual rejection.
		const harness = await createHarness({
			models: [{ id: "faux-1", contextWindow: 0 }],
			tools: [],
		});
		harnesses.push(harness);
		harness.setResponses([fauxAssistantMessage("ok")]);
		await harness.session.prompt("hi");
		expect(harness.session.agent.state.errorMessage).toBeUndefined();
	});
});
