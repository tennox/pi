/**
 * TuiMainScreen ("regular" mode) widget-churn benchmark.
 *
 * This is the default pi interactive mode (not the alt-screen/fullscreen mode covered by
 * alt-screen-large-transcript-bench.ts). A live idle session profile showed 85.5% of process CPU
 * in the render path driven by an extension widget's 80ms spinner tick: each tick calls
 * `ui.requestRender()`, which triggers a FULL `doRender()` even though only one small widget
 * changed. The cost scaled with total scrollback size, not with what actually changed.
 *
 * Layout mirrors default interactive mode: TuiMainScreen's children are appended directly (no
 * viewport/ScrollView - "regular" mode just grows the terminal's native scrollback):
 *
 * TuiMainScreen
 * ├── chat Container
 * │   └── N message components, alternating:
 * │       - assistant Markdown message (representative prose)
 * │       - bash-tool-result preview (Container [ header Text, cached preview leaf ]) with a
 * │         "... (N earlier lines, ctrl+o to expand)" hint - the single biggest hot path in the
 * │         profile (truncateToWidth on this hint reached 28.2% of total CPU before the leaf's
 * │         result array was memoized).
 * ├── editor Text (static)
 * └── belowEditor spinner widget (content changes every frame)
 *
 * The headline result: per-frame cost should scale with the CHANGED region (the one-line
 * spinner), not with scrollback size. This is demonstrated by running the same "steady spinner
 * churn" scenario at two scrollback sizes and comparing ms/frame - before the fix this ratio
 * tracks the size ratio; after the fix it should be close to 1.
 *
 * Run from the repository root:
 *   node --experimental-strip-types packages/tui/test/main-screen-widget-churn-bench.ts
 */

import { Session } from "node:inspector/promises";
import { performance } from "node:perf_hooks";
import { Markdown, type MarkdownTheme } from "../src/components/markdown.ts";
import { Text } from "../src/components/text.ts";
import type { Terminal } from "../src/terminal.ts";
import { type Component, Container } from "../src/tui.ts";
import { TuiMainScreen } from "../src/tui-main-screen.ts";
import { truncateToWidth, wrapTextWithAnsi } from "../src/utils.ts";

const COLUMNS = 100;
const ROWS = 40;
const WARMUP_FRAMES = 20;
const FRAMES = 300;
const SAMPLING_INTERVAL = 4096;
const SCROLLBACK_SIZES = [500, 4_000];

/** Terminal that discards output; keeps ANSI/terminal parsing out of the measurement. */
class NullTerminal implements Terminal {
	columns = COLUMNS;
	rows = ROWS;

	start(_onInput: (data: string) => void, _onResize: () => void): void {}
	stop(): void {}
	async drainInput(): Promise<void> {}
	write(_data: string): void {}
	get kittyProtocolActive(): boolean {
		return false;
	}
	moveBy(_lines: number): void {}
	hideCursor(): void {}
	showCursor(): void {}
	clearLine(): void {}
	clearFromCursor(): void {}
	clearScreen(): void {}
	setTitle(_title: string): void {}
	setProgress(_active: boolean): void {}
}

const markdownTheme: MarkdownTheme = {
	heading: (text) => text,
	link: (text) => text,
	linkUrl: (text) => text,
	code: (text) => text,
	codeBlock: (text) => text,
	codeBlockBorder: (text) => text,
	quote: (text) => text,
	quoteBorder: (text) => text,
	hr: (text) => text,
	listBullet: (text) => text,
	bold: (text) => text,
	italic: (text) => text,
	strikethrough: (text) => text,
	underline: (text) => text,
};

const assistantMarkdown = [
	"## Analysis",
	"",
	"Inspected the implementation and confirmed the observed behavior. The relevant code path " +
		"reads the cached state, applies the requested transform, and writes the result back " +
		"without additional allocation in the common case.",
	"",
	"- preserves correctness under concurrent updates",
	"- avoids reallocating the backing buffer on the hot path",
	"- keeps the public API unchanged",
	"",
	"```ts",
	"function apply(state: State, update: Update): State {",
	"  if (state.version !== update.baseVersion) throw new Error('stale update');",
	"  return { ...state, ...update.patch, version: state.version + 1 };",
	"}",
	"```",
].join("\n");

const BASH_PREVIEW_LINES = 5;

/**
 * Minimal stand-in for the coding-agent bash tool result renderer
 * (packages/coding-agent/src/core/tools/renderers/bash.ts). It reproduces the exact shape of the
 * profiled hot path: a collapsed preview with a styled "... (N earlier lines, to expand)" hint
 * that is truncated to width, prepended to a handful of cached preview lines. The real renderer
 * memoizes the fully assembled array by width (Step 2 of this fix); this benchmark component
 * mirrors that fixed shape so the surrounding Container/TuiMainScreen fixes (Steps 3-4) can be
 * measured against a realistic, already-cached leaf.
 */
class BashResultPreview implements Component {
	private cachedWidth: number | undefined;
	private cachedResult: string[] | undefined;
	private readonly styledOutput: string;
	private readonly skippedCount: number;

	constructor(styledOutput: string, skippedCount: number) {
		this.styledOutput = styledOutput;
		this.skippedCount = skippedCount;
	}

	invalidate(): void {
		this.cachedWidth = undefined;
		this.cachedResult = undefined;
	}

	render(width: number): string[] {
		if (this.cachedResult !== undefined && this.cachedWidth === width) {
			return this.cachedResult;
		}
		const previewLines = wrapTextWithAnsi(this.styledOutput, width).slice(0, BASH_PREVIEW_LINES);
		const hint = `\x1b[2m... (${this.skippedCount} earlier lines,\x1b[22m \x1b[36mctrl+o\x1b[39m\x1b[2m to expand)\x1b[22m`;
		const result = ["", truncateToWidth(hint, width, "..."), ...previewLines];
		this.cachedWidth = width;
		this.cachedResult = result;
		return result;
	}
}

const bashCommandOutput = [
	"drwxr-xr-x  12 dev  staff   384 Sep 15 10:02 .",
	"drwxr-xr-x   8 dev  staff   256 Sep 14 09:11 ..",
	"-rw-r--r--   1 dev  staff  1821 Sep 15 09:58 \x1b[32mindex.ts\x1b[39m",
	"-rw-r--r--   1 dev  staff  4210 Sep 15 09:58 \x1b[32mutils.ts\x1b[39m",
	"drwxr-xr-x   4 dev  staff   128 Sep 15 09:40 \x1b[34mcomponents\x1b[39m",
	"drwxr-xr-x   6 dev  staff   192 Sep 15 09:40 \x1b[34mtest\x1b[39m",
	"-rw-r--r--   1 dev  staff   612 Sep 12 08:20 package.json",
	"-rw-r--r--   1 dev  staff  2044 Sep 12 08:20 tsconfig.json",
	"-rw-r--r--   1 dev  staff   890 Sep 10 17:03 README.md",
	"-rw-r--r--   1 dev  staff  1590 Sep 15 08:12 CHANGELOG.md",
	"-rw-r--r--   1 dev  staff   413 Sep  9 14:55 .gitignore",
	"drwxr-xr-x   3 dev  staff    96 Sep  8 11:30 \x1b[34mdist\x1b[39m",
].join("\n");

/** Extension widget stand-in: an 80ms spinner tick that always returns fresh content. */
class SpinnerWidget implements Component {
	private static readonly FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
	private tick = 0;

	next(): void {
		this.tick++;
	}

	invalidate(): void {}

	render(_width: number): string[] {
		const frame = SpinnerWidget.FRAMES[this.tick % SpinnerWidget.FRAMES.length];
		return [`${frame} Working... (${this.tick}s)`];
	}
}

interface Scenario {
	tui: TuiMainScreen;
	terminal: NullTerminal;
	spinner: SpinnerWidget;
	totalLines: number;
}

function buildScenario(componentCount: number): Scenario {
	const chat = new Container();
	for (let index = 0; index < componentCount; index++) {
		if (index % 4 === 3) {
			const bashResult = new Container();
			bashResult.addChild(new Text(`\x1b[1m$ ls -la\x1b[22m`, 0, 0));
			bashResult.addChild(new BashResultPreview(bashCommandOutput, 40 + (index % 17)));
			chat.addChild(bashResult);
		} else {
			chat.addChild(new Markdown(assistantMarkdown, 1, 0, markdownTheme));
		}
	}

	const editor = new Text("\x1b[90m─\x1b[39m".repeat(1) + "\n > \n" + "\x1b[90m─\x1b[39m", 0, 0);
	const spinner = new SpinnerWidget();

	const terminal = new NullTerminal();
	const tui = new TuiMainScreen(terminal, false, "/tmp/pi-tui-bench");
	tui.addChild(chat);
	tui.addChild(editor);
	tui.addChild(spinner);
	tui.start();
	tui.renderNow();

	const totalLines = chat.render(terminal.columns).length;

	for (let frame = 0; frame < WARMUP_FRAMES; frame++) {
		spinner.next();
		tui.renderNow();
	}

	return { tui, terminal, spinner, totalLines };
}

interface FrameResult {
	msPerFrame: number;
	kibPerFrame: number;
}

async function measureSpinnerChurn(scenario: Scenario): Promise<FrameResult> {
	const session = new Session();
	session.connect();
	await session.post("HeapProfiler.startSampling", {
		samplingInterval: SAMPLING_INTERVAL,
		includeObjectsCollectedByMajorGC: true,
		includeObjectsCollectedByMinorGC: true,
	});

	const start = performance.now();
	for (let frame = 0; frame < FRAMES; frame++) {
		scenario.spinner.next();
		scenario.tui.renderNow();
	}
	const elapsedMs = performance.now() - start;

	const { profile } = await session.post("HeapProfiler.stopSampling");
	session.disconnect();

	interface SamplingNode {
		selfSize: number;
		children: SamplingNode[];
	}
	const sumProfile = (node: SamplingNode): number => {
		let total = node.selfSize;
		for (const child of node.children) total += sumProfile(child);
		return total;
	};
	const allocatedBytes = sumProfile(profile.head as SamplingNode);

	scenario.tui.stop();

	return {
		msPerFrame: elapsedMs / FRAMES,
		kibPerFrame: allocatedBytes / FRAMES / 1024,
	};
}

async function main(): Promise<void> {
	const results: Array<{ size: number; totalLines: number; result: FrameResult }> = [];

	for (const size of SCROLLBACK_SIZES) {
		const scenario = buildScenario(size);
		const totalLines = scenario.totalLines;
		const result = await measureSpinnerChurn(scenario);
		results.push({ size, totalLines, result });
		console.log(
			`components=${String(size).padStart(5)}  lines=${String(totalLines).padStart(7)}  ` +
				`spinner-churn: ${result.msPerFrame.toFixed(3).padStart(7)} ms/frame  ` +
				`${result.kibPerFrame.toFixed(1).padStart(8)} KiB/frame  (${FRAMES} frames)`,
		);
	}

	const first = results[0]!;
	const last = results[results.length - 1]!;
	const sizeRatio = last.size / first.size;
	const msRatio = last.result.msPerFrame / first.result.msPerFrame;
	console.log("");
	console.log(
		`scrollback size ratio: ${sizeRatio.toFixed(1)}x  ->  ms/frame ratio: ${msRatio.toFixed(2)}x ` +
			"(should be close to 1x if per-frame cost is O(changed region), not O(scrollback))",
	);
}

await main();
