/**
 * Regression tests for the bash tool result renderer's collapsed-preview cache
 * (packages/coding-agent/src/core/tools/renderers/bash.ts).
 *
 * A V8 profile of an idle pi session found 28.2% of total process CPU flowing through this
 * renderer's collapsed-preview leaf: it cached the truncated output lines by width, but rebuilt
 * the "... (N earlier lines, to expand)" hint and reallocated the returned array on every single
 * render call, even when nothing had changed. With many bash results in scrollback rendered at
 * ~12.5fps (an 80ms spinner tick), that per-frame rebuild dominated the profile.
 *
 * The fix memoizes the fully assembled array by width so a repeat render at the same width
 * returns the identical array reference (required for Container-level caching further up the
 * tree). These tests assert: (1) the cache returns a stable reference and correct content across
 * repeated renders, (2) a width change forces recomputation with correct truncation, (3)
 * `invalidate()` (the theme/keybinding-change signal) forces recomputation, and (4) toggling
 * `options.expanded` (the Ctrl+O collapse/expand action) still produces the correct output.
 */
import assert from "node:assert";
import type { Component } from "@earendil-works/pi-tui";
import { beforeAll, describe, it } from "vitest";
import type { ToolRenderContext, ToolRenderResultOptions } from "../src/core/extensions/types.ts";
import { createShellRenderers } from "../src/core/tools/renderers/bash.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";

function makeContext(overrides: Partial<ToolRenderContext> = {}): ToolRenderContext {
	return {
		args: { command: "ls" },
		toolCallId: "call-1",
		invalidate: () => {},
		lastComponent: undefined,
		state: {},
		cwd: "/tmp",
		executionStarted: true,
		argsComplete: true,
		isPartial: false,
		expanded: false,
		showImages: false,
		isError: false,
		...overrides,
	};
}

function makeResult(text: string): { content: Array<{ type: string; text: string }> } {
	return { content: [{ type: "text", text }] };
}

function makeOptions(overrides: Partial<ToolRenderResultOptions> = {}): ToolRenderResultOptions {
	return { expanded: false, isPartial: false, ...overrides };
}

/** Many lines so the collapsed preview always has a non-empty "earlier lines" hint. */
const longOutput = Array.from({ length: 20 }, (_, i) => `line ${i}`).join("\n");

describe("bash result renderer collapsed-preview cache", () => {
	beforeAll(() => {
		initTheme(undefined, false);
	});

	function renderCollapsed(
		width: number,
		context: ToolRenderContext,
		isPartial = false,
	): { component: Component; lines: string[] } {
		const renderers = createShellRenderers("$");
		const component = renderers.renderResult!(
			makeResult(longOutput) as any,
			makeOptions({ isPartial }),
			{} as any,
			context,
		);
		return { component, lines: component.render(width) };
	}

	it("returns the identical array reference for a repeat render at the same width", () => {
		const context = makeContext();
		const renderers = createShellRenderers("$");
		const component = renderers.renderResult!(makeResult(longOutput) as any, makeOptions(), {} as any, context);

		const first = component.render(80);
		const second = component.render(80);
		assert.strictEqual(first, second, "expected the same array reference on a width-stable repeat render");
		assert.ok(
			first.some((line) => line.includes("earlier lines")),
			"expected an earlier-lines hint",
		);
	});

	it("recomputes with correct truncation after a width change", () => {
		const context = makeContext();
		const renderers = createShellRenderers("$");
		const component = renderers.renderResult!(makeResult(longOutput) as any, makeOptions(), {} as any, context);

		const wide = component.render(80);
		const narrow = component.render(40);
		assert.notStrictEqual(wide, narrow, "expected a new array after a width change");
		for (const line of narrow) {
			// Strip ANSI escapes before measuring; a plain length check catches gross truncation bugs.
			const stripped = line.replace(/\x1b\[[0-9;]*m/g, "");
			assert.ok(stripped.length <= 40, `line exceeds narrow width: ${JSON.stringify(stripped)}`);
		}

		// Re-widening returns to the original (recomputed, but content-equal) output.
		const wideAgain = component.render(80);
		assert.deepStrictEqual(wideAgain, wide);
	});

	it("recomputes after invalidate() (theme/keybinding change signal)", () => {
		const context = makeContext();
		const { component, lines: before } = renderCollapsed(80, context);

		component.invalidate();
		const after = component.render(80);

		assert.notStrictEqual(before, after, "expected a new array reference after invalidate()");
		assert.deepStrictEqual(before, after, "content should be identical since nothing else changed");
	});

	it("switches between collapsed preview and full expanded output (Ctrl+O)", () => {
		const context = makeContext();
		const renderers = createShellRenderers("$");

		const collapsedComponent = renderers.renderResult!(
			makeResult(longOutput) as any,
			makeOptions({ expanded: false }),
			{} as any,
			context,
		);
		const collapsedLines = collapsedComponent.render(80);
		assert.ok(collapsedLines.some((line) => line.includes("earlier lines")));
		assert.ok(!collapsedLines.some((line) => line.includes("line 0")), "collapsed view should omit earlier lines");

		// Ctrl+O toggles expanded=true and re-renders onto the *same* component instance
		// (interactive-mode passes context.lastComponent back in).
		const expandedComponent = renderers.renderResult!(
			makeResult(longOutput) as any,
			makeOptions({ expanded: true }),
			{} as any,
			makeContext({ lastComponent: collapsedComponent }),
		);
		const expandedLines = expandedComponent.render(80);
		assert.ok(
			expandedLines.some((line) => line.includes("line 19")),
			"expanded view should include all lines",
		);
		assert.ok(!expandedLines.some((line) => line.includes("earlier lines")));

		// Collapsing again re-populates a fresh collapsed cache correctly.
		const recollapsedComponent = renderers.renderResult!(
			makeResult(longOutput) as any,
			makeOptions({ expanded: false }),
			{} as any,
			makeContext({ lastComponent: expandedComponent }),
		);
		const recollapsedLines = recollapsedComponent.render(80);
		assert.deepStrictEqual(recollapsedLines, collapsedLines);
	});

	it("keeps a streaming (isPartial) update to the tail line reflected on re-render", () => {
		const context = makeContext({ isPartial: true });
		const { component: c1, lines: firstLines } = renderCollapsed(80, context, true);
		assert.ok(firstLines.length > 0);

		// A later streaming update rebuilds the component's children (rebuildBashResultRenderComponent
		// clears and re-adds), so the cache must reflect the newest output, not stale content.
		const renderers = createShellRenderers("$");
		const updatedComponent = renderers.renderResult!(
			makeResult(`${longOutput}\nline 20 (streamed)`) as any,
			makeOptions({ isPartial: true }),
			{} as any,
			makeContext({ isPartial: true, lastComponent: c1 }),
		);
		const updatedLines = updatedComponent.render(80);
		assert.notDeepStrictEqual(updatedLines, firstLines);
	});
});
