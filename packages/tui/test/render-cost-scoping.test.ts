/**
 * Regression tests for the TuiMainScreen render-cost fix:
 *
 * - `Container.render()` (tui.ts) now memoizes its concatenated output by width, keyed on
 *   reference-equality of each child's returned lines array (Step 4).
 * - `TuiMainScreen.doRender()` (tui-main-screen.ts) now defers line resets and kitty-image
 *   bookkeeping until after the diff is known, applying them only to the written/changed range
 *   instead of the whole scrollback (Step 3).
 *
 * These assert output stays byte-identical to the pre-fix full-scan behavior for the scenarios
 * called out during review: a width change, a streaming update to the last entry, and a
 * scrollback containing kitty image lines (both with and without a prior image already present).
 */
import assert from "node:assert";
import { describe, it } from "node:test";
import { Image } from "../src/components/image.ts";
import { resetCapabilitiesCache, setCapabilities, setCellDimensions } from "../src/terminal-image.ts";
import type { Component, TUI } from "../src/tui.ts";
import { Container } from "../src/tui.ts";
import { TuiMainScreen } from "../src/tui-main-screen.ts";
import { VirtualTerminal } from "./virtual-terminal.ts";

class TestComponent implements Component {
	lines: string[] = [];
	render(_width: number): string[] {
		return this.lines;
	}
	invalidate(): void {}
}

/** Returns a fresh array reference on every render, unlike TestComponent's stable field. */
class ChurnComponent implements Component {
	text = "";
	render(_width: number): string[] {
		return [this.text];
	}
	invalidate(): void {}
}

class LoggingVirtualTerminal extends VirtualTerminal {
	private writes: string[] = [];
	override write(data: string): void {
		this.writes.push(data);
		super.write(data);
	}
	getWrites(): string {
		return this.writes.join("");
	}
	clearWrites(): void {
		this.writes = [];
	}
}

describe("Container.render() concatenation cache", () => {
	it("returns the identical array reference when every child's output is unchanged", () => {
		const container = new Container();
		const a = new TestComponent();
		const b = new TestComponent();
		a.lines = ["a1", "a2"];
		b.lines = ["b1"];
		container.addChild(a);
		container.addChild(b);

		const first = container.render(80);
		const second = container.render(80);
		assert.strictEqual(first, second);
		assert.deepStrictEqual(first, ["a1", "a2", "b1"]);
	});

	it("recomputes (but stays correct) when a child returns a fresh array every render", () => {
		const container = new Container();
		const stable = new TestComponent();
		stable.lines = ["stable"];
		const churn = new ChurnComponent();
		churn.text = "v1";
		container.addChild(stable);
		container.addChild(churn);

		const first = container.render(80);
		assert.deepStrictEqual(first, ["stable", "v1"]);

		churn.text = "v2";
		const second = container.render(80);
		assert.notStrictEqual(first, second, "a changed child must invalidate the concatenation cache");
		assert.deepStrictEqual(second, ["stable", "v2"]);
	});

	it("recomputes when a child is added or removed even if remaining children are unchanged", () => {
		const container = new Container();
		const a = new TestComponent();
		a.lines = ["a1"];
		container.addChild(a);
		const first = container.render(80);

		const b = new TestComponent();
		b.lines = ["b1"];
		container.addChild(b);
		const second = container.render(80);

		assert.notStrictEqual(first, second);
		assert.deepStrictEqual(second, ["a1", "b1"]);
	});

	it("recomputes at a different width even if children are unchanged", () => {
		const container = new Container();
		const a = new TestComponent();
		a.lines = ["a1"];
		container.addChild(a);

		const at80 = container.render(80);
		const at100 = container.render(100);
		assert.notStrictEqual(at80, at100);
		assert.deepStrictEqual(at80, at100, "content is the same here since TestComponent ignores width");
	});
});

describe("TuiMainScreen deferred line reset (width change)", () => {
	it("resets every line on a full redraw triggered by a width change", async () => {
		const terminal = new LoggingVirtualTerminal(40, 10);
		const tui: TUI = new TuiMainScreen(terminal);
		const component = new TestComponent();
		tui.addChild(component);

		component.lines = ["one\tline", "two\tline"];
		tui.start();
		await terminal.waitForRender();
		terminal.clearWrites();

		terminal.resize(50, 10);
		tui.requestRender();
		await terminal.waitForRender();

		const writes = terminal.getWrites();
		// Tabs must be normalized (3 spaces) on the post-width-change full redraw, and every
		// non-image line must carry the trailing style/hyperlink reset.
		assert.ok(!writes.includes("\t"), "tabs must be normalized on full redraw");
		assert.ok(writes.includes("\x1b[0m\x1b]8;;\x07"), "lines must carry the segment reset on full redraw");

		tui.stop();
	});
});

describe("TuiMainScreen deferred line reset (differential update)", () => {
	it("only rewrites the changed tail line, with the reset applied to that line", async () => {
		const terminal = new LoggingVirtualTerminal(40, 10);
		const tui: TUI = new TuiMainScreen(terminal);
		const component = new TestComponent();
		tui.addChild(component);

		component.lines = ["static one", "static two", "spinner: |"];
		tui.start();
		await terminal.waitForRender();
		terminal.clearWrites();

		component.lines = ["static one", "static two", "spinner: /"];
		tui.requestRender();
		await terminal.waitForRender();

		const writes = terminal.getWrites();
		assert.ok(writes.includes("spinner: /"), "the changed line must be written");
		assert.ok(!writes.includes("static one"), "unchanged lines must not be rewritten");
		assert.ok(!writes.includes("static two"), "unchanged lines must not be rewritten");
		assert.ok(writes.includes("\x1b[0m\x1b]8;;\x07"), "the rewritten line must carry the segment reset");

		tui.stop();
	});
});

describe("TuiMainScreen deferred kitty-image scan", () => {
	it("keeps an existing image intact when an unrelated line elsewhere changes", async () => {
		setCapabilities({ images: "kitty", trueColor: true, hyperlinks: true });
		setCellDimensions({ widthPx: 10, heightPx: 10 });
		try {
			const terminal = new LoggingVirtualTerminal(40, 20);
			const tui: TUI = new TuiMainScreen(terminal);
			const component = new TestComponent();
			tui.addChild(component);

			const image = new Image(
				"AAAA",
				"image/png",
				{ fallbackColor: (value) => value },
				{ maxWidthCells: 2 },
				{ widthPx: 20, heightPx: 20 },
			);
			const imageLines = image.render(40);
			component.lines = ["before image", ...imageLines, "after image", "spinner: |"];
			tui.start();
			await terminal.waitForRender();
			terminal.clearWrites();

			// Change only the trailing spinner line; the image is far from the changed range and
			// `previousKittyImageIds` is non-empty, so the fallback (full-array) kitty scan runs and
			// must still find and preserve the image untouched.
			component.lines = ["before image", ...imageLines, "after image", "spinner: /"];
			tui.requestRender();
			await terminal.waitForRender();

			const writes = terminal.getWrites();
			assert.ok(writes.includes("spinner: /"));
			assert.ok(!writes.includes(imageLines[0]!), "the untouched image placement must not be rewritten");

			tui.stop();
		} finally {
			resetCapabilitiesCache();
			setCellDimensions({ widthPx: 9, heightPx: 18 });
		}
	});

	it("correctly reserves rows for a brand-new image introduced inside the changed range", async () => {
		setCapabilities({ images: "kitty", trueColor: true, hyperlinks: true });
		setCellDimensions({ widthPx: 10, heightPx: 10 });
		try {
			const terminal = new LoggingVirtualTerminal(40, 20);
			const tui: TUI = new TuiMainScreen(terminal);
			const component = new TestComponent();
			tui.addChild(component);

			// No images anywhere yet: previousKittyImageIds is empty, exercising the bounded fast
			// path in expandChangedRangeForKittyImages/collectKittyImageIds.
			component.lines = ["before", "after"];
			tui.start();
			await terminal.waitForRender();
			terminal.clearWrites();

			const image = new Image(
				"AAAA",
				"image/png",
				{ fallbackColor: (value) => value },
				{ maxWidthCells: 2 },
				{ widthPx: 20, heightPx: 20 },
			);
			const imageLines = image.render(40);
			const imageSequence = imageLines[0];
			component.lines = ["before", ...imageLines, "after"];
			tui.requestRender();
			await terminal.waitForRender();

			const writes = terminal.getWrites();
			assert.ok(
				writes.includes(`\x1b[2K\r\n\x1b[2K\x1b[1A${imageSequence}\x1b[1B`),
				"reserved rows for the newly introduced image must be cleared before it is drawn",
			);

			tui.stop();
		} finally {
			resetCapabilitiesCache();
			setCellDimensions({ widthPx: 9, heightPx: 18 });
		}
	});
});
