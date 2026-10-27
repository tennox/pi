/**
 * Tests for the per-response parallel tool-call cap in agent-loop.ts.
 *
 * Background: a single assistant response can carry thousands of tool calls
 * (one observed incident: 2,454 calls, ~538 KB of arguments, ~2.2M tokens of
 * results hitting a 1M-token window → provider 400 → agent died empty-handed).
 * The cap runs at the start of tool-call execution. Excess calls are converted
 * to immediate error tool results with a visible "why" message so the model
 * sees the cap and can recover on the next turn.
 */

import {
	type AssistantMessage,
	createAssistantMessageEventStream,
	type Message,
	type Model,
	type UserMessage,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { agentLoop } from "../src/agent-loop.ts";
import type { AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool } from "../src/types.ts";

// ---------------------------------------------------------------------------
// Helpers (kept minimal; the parent test file's helpers are not exported).
// ---------------------------------------------------------------------------

function createUsage() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function createModel(): Model<"openai-responses"> {
	return {
		id: "mock",
		name: "mock",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "https://example.invalid",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8192,
		maxTokens: 2048,
	};
}

function createAssistantMessage(
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "openai-responses",
		provider: "openai",
		model: "mock",
		usage: createUsage(),
		stopReason,
		timestamp: Date.now(),
	};
}

function createUserMessage(text: string): UserMessage {
	return {
		role: "user",
		content: text,
		timestamp: Date.now(),
	};
}

function identityConverter(messages: AgentMessage[]): Message[] {
	return messages.filter(
		(m) => m.role === "system" || m.role === "user" || m.role === "assistant" || m.role === "toolResult",
	) as Message[];
}

function getToolResultText(message: AgentMessage): string {
	if (message.role !== "toolResult") return "";
	return message.content
		.filter((part): part is { type: "text"; text: string } => part.type === "text")
		.map((part) => part.text)
		.join("\n");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("parallel tool-call cap (PI_MAX_PARALLEL_TOOL_CALLS)", () => {
	// Restore any env overrides between tests so tests do not bleed.
	const originalMaxCalls = process.env.PI_MAX_PARALLEL_TOOL_CALLS;
	const originalMaxBytes = process.env.PI_MAX_PARALLEL_TOOL_CALLS_ARG_BYTES;
	beforeEach(() => {
		delete process.env.PI_MAX_PARALLEL_TOOL_CALLS;
		delete process.env.PI_MAX_PARALLEL_TOOL_CALLS_ARG_BYTES;
	});
	afterEach(() => {
		if (originalMaxCalls === undefined) delete process.env.PI_MAX_PARALLEL_TOOL_CALLS;
		else process.env.PI_MAX_PARALLEL_TOOL_CALLS = originalMaxCalls;
		if (originalMaxBytes === undefined) delete process.env.PI_MAX_PARALLEL_TOOL_CALLS_ARG_BYTES;
		else process.env.PI_MAX_PARALLEL_TOOL_CALLS_ARG_BYTES = originalMaxBytes;
	});

	it("executes only the first 20 of 2,454 parallel tool calls and emits a visible drop note for the rest", async () => {
		const executed: string[] = [];
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};
		const context: AgentContext = { messages: [], tools: [tool] };
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		const TOTAL = 2454;
		const EXPECTED_CAP = 20;
		const toolCalls = Array.from({ length: TOTAL }, (_, i) => ({
			type: "toolCall" as const,
			id: `tc-${i}`,
			name: "echo",
			arguments: { value: `v${i}` },
		}));

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("do a lot")], context, config, undefined, () => {
			llmCalls++;
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					stream.push({
						type: "done",
						reason: "toolUse",
						message: createAssistantMessage(toolCalls, "toolUse"),
					});
				} else {
					stream.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: "ok" }]),
					});
				}
			});
			return stream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		// Only the first 20 calls reached the tool's execute().
		expect(executed).toHaveLength(EXPECTED_CAP);
		for (let i = 0; i < EXPECTED_CAP; i++) {
			expect(executed[i]).toBe(`v${i}`);
		}

		// Every dropped call surfaces as a tool_execution_end (isError: true) and as a
		// toolResult message carrying the visible "why".
		const toolEnds = events.filter((e) => e.type === "tool_execution_end");
		expect(toolEnds).toHaveLength(TOTAL);
		for (const end of toolEnds) {
			if (end.type !== "tool_execution_end") continue;
			if (end.isError) {
				expect(end.result.content.find((c: { type: string }) => c.type === "text")?.text).toContain(
					"per-response call cap of 20 calls",
				);
			}
		}

		// The transcript contains toolResult messages for ALL 2,454 calls; the model
		// sees the cap rather than getting no feedback at all.
		const messages = await stream.result();
		const toolResults = messages.filter((m) => m.role === "toolResult");
		expect(toolResults).toHaveLength(TOTAL);
		const droppedResults = toolResults.filter((m) => m.role === "toolResult" && m.isError);
		expect(droppedResults).toHaveLength(TOTAL - EXPECTED_CAP);
		for (const dropped of droppedResults) {
			if (dropped.role !== "toolResult") continue;
			expect(getToolResultText(dropped)).toContain("not executed");
			expect(getToolResultText(dropped)).toContain("per-response call cap");
		}

		// The loop continues so the model can recover on the next response.
		expect(llmCalls).toBe(2);
	});

	it("trips the cumulative arg-byte cap and drops calls past the budget with a visible reason", async () => {
		const executed: number[] = [];
		// Build a single oversized argument blob: ~10 KB per call.
		const bigValue = "x".repeat(10_000);
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value.length);
				return {
					content: [{ type: "text", text: `echoed: ${params.value.length}` }],
					details: { value: String(params.value.length) },
				};
			},
		};
		const context: AgentContext = { messages: [], tools: [tool] };
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		// 30 calls of ~10 KB args each ≈ 300 KB total; default budget is 128 KB.
		const TOTAL = 30;
		const toolCalls = Array.from({ length: TOTAL }, (_, i) => ({
			type: "toolCall" as const,
			id: `tc-${i}`,
			name: "echo",
			arguments: { value: `${i}-${bigValue}` },
		}));

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("big batch")], context, config, undefined, () => {
			llmCalls++;
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					stream.push({
						type: "done",
						reason: "toolUse",
						message: createAssistantMessage(toolCalls, "toolUse"),
					});
				} else {
					stream.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: "ok" }]),
					});
				}
			});
			return stream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		// Some calls executed (we don't pin the exact count — it depends on JSON
		// framing). The contract is: total drops > 0 AND the budget reason appears.
		const toolEnds = events.filter((e) => e.type === "tool_execution_end");
		expect(toolEnds).toHaveLength(TOTAL);
		const dropped = toolEnds.filter((e) => e.type === "tool_execution_end" && e.isError);
		expect(dropped.length).toBeGreaterThan(0);
		expect(executed.length).toBeLessThan(TOTAL);
		// At least one dropped call mentions the byte budget (the first 20 calls go
		// past the count cap so drops beyond index 20 are tagged as count-cap, but
		// calls past the budget inside the first 20 must surface the byte reason).
		const budgetDrops = dropped.filter((e) => {
			if (e.type !== "tool_execution_end") return false;
			const text = e.result.content.find((c: { type: string }) => c.type === "text")?.text ?? "";
			return text.includes("argument-byte budget");
		});
		expect(budgetDrops.length).toBeGreaterThan(0);
	});

	it("honors a smaller env override (PI_MAX_PARALLEL_TOOL_CALLS=3)", async () => {
		process.env.PI_MAX_PARALLEL_TOOL_CALLS = "3";
		const executed: string[] = [];
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};
		const context: AgentContext = { messages: [], tools: [tool] };
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		const toolCalls = Array.from({ length: 10 }, (_, i) => ({
			type: "toolCall" as const,
			id: `tc-${i}`,
			name: "echo",
			arguments: { value: `v${i}` },
		}));

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("cap me")], context, config, undefined, () => {
			llmCalls++;
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					stream.push({
						type: "done",
						reason: "toolUse",
						message: createAssistantMessage(toolCalls, "toolUse"),
					});
				} else {
					stream.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: "ok" }]),
					});
				}
			});
			return stream;
		});
		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}
		expect(executed).toHaveLength(3);
		const messages = await stream.result();
		const toolResults = messages.filter((m) => m.role === "toolResult");
		expect(toolResults).toHaveLength(10);
		const dropped = toolResults.filter((m) => m.role === "toolResult" && m.isError);
		expect(dropped).toHaveLength(7);
		for (const d of dropped) {
			if (d.role !== "toolResult") continue;
			expect(getToolResultText(d)).toContain("per-response call cap of 3 calls");
		}
	});

	it("leaves a small batch (5 of 20 calls) untouched — no drops, no synthetic notes", async () => {
		const executed: string[] = [];
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};
		const context: AgentContext = { messages: [], tools: [tool] };
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		const toolCalls = Array.from({ length: 5 }, (_, i) => ({
			type: "toolCall" as const,
			id: `tc-${i}`,
			name: "echo",
			arguments: { value: `v${i}` },
		}));

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("normal batch")], context, config, undefined, () => {
			llmCalls++;
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					stream.push({
						type: "done",
						reason: "toolUse",
						message: createAssistantMessage(toolCalls, "toolUse"),
					});
				} else {
					stream.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: "ok" }]),
					});
				}
			});
			return stream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		expect(executed).toEqual(["v0", "v1", "v2", "v3", "v4"]);
		const toolEnds = events.filter((e) => e.type === "tool_execution_end");
		for (const end of toolEnds) {
			if (end.type !== "tool_execution_end") continue;
			expect(end.isError).toBe(false);
		}
		const messages = await stream.result();
		const toolResults = messages.filter((m) => m.role === "toolResult");
		expect(toolResults).toHaveLength(5);
		for (const r of toolResults) {
			if (r.role !== "toolResult") continue;
			expect(r.isError).toBe(false);
		}
	});
});
