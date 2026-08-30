import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import St from 'gi://St';

import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';

const STATE_PATH = '/home/echidna/heiwa_v2/state/embodied_runtime/cursor_guidance_overlay.json';
const POINTER_PATH = '/home/echidna/heiwa_v2/state/embodied_runtime/shell_pointer.json';

export default class HeiwaCursorOverlay extends Extension {
    enable() {
        this._root = new St.Widget({
            style_class: 'heiwa-cursor-overlay',
            reactive: false,
            can_focus: false,
        });
        Main.layoutManager.addTopChrome(this._root, {trackFullscreen: true});
        this._monitorId = Main.layoutManager.connect('monitors-changed', () => this._place());
        this._place();
        this._refresh();
        this._timer = GLib.timeout_add(GLib.PRIORITY_DEFAULT, 100, () => {
            this._refresh();
            return GLib.SOURCE_CONTINUE;
        });
    }

    disable() {
        if (this._timer) {
            GLib.source_remove(this._timer);
            this._timer = 0;
        }
        if (this._monitorId) {
            Main.layoutManager.disconnect(this._monitorId);
            this._monitorId = 0;
        }
        if (this._root) {
            Main.layoutManager.removeChrome(this._root);
            this._root.destroy();
            this._root = null;
        }
    }

    _place() {
        const monitor = Main.layoutManager.primaryMonitor;
        if (!monitor || !this._root)
            return;
        this._root.set_position(monitor.x, monitor.y);
        this._root.set_size(monitor.width, monitor.height);
    }

    _box(styleClass, x, y, width, height) {
        const box = new St.Widget({style_class: styleClass, reactive: false});
        box.set_position(Math.round(x), Math.round(y));
        box.set_size(Math.max(Math.round(width), 2), Math.max(Math.round(height), 2));
        this._root.add_child(box);
    }

    _cell(styleClass, cell, gw, gh, width, height) {
        const row = Math.floor(cell / gw);
        const col = cell % gw;
        const cw = width / gw;
        const ch = height / gh;
        this._box(styleClass, col * cw + 2, row * ch + 2, cw - 4, ch - 4);
    }

    _refresh() {
        if (!this._root)
            return;
        let payload;
        try {
            const file = Gio.File.new_for_path(STATE_PATH);
            const [ok, contents] = file.load_contents(null);
            if (!ok)
                return;
            payload = JSON.parse(new TextDecoder().decode(contents));
        } catch (_error) {
            return;
        }
        if (payload.format !== 1 || !Array.isArray(payload.grid))
            return;
        this._publishPointer();
        this._root.destroy_all_children();
        const width = this._root.width;
        const height = this._root.height;
        const [gw, gh] = payload.grid.map(Number);
        if (!(gw > 0 && gh > 0 && width > 0 && height > 0))
            return;

        for (const item of payload.top_candidates ?? [])
            this._cell('heiwa-candidate', Number(item.cell), gw, gh, width, height);
        for (const cell of payload.rejected_cells ?? [])
            this._cell('heiwa-rejected', Number(cell), gw, gh, width, height);
        for (const cell of payload.hard_rejected_cells ?? [])
            this._cell('heiwa-hard-rejected', Number(cell), gw, gh, width, height);
        for (const cell of payload.attempted_cells ?? [])
            this._cell('heiwa-attempted', Number(cell), gw, gh, width, height);
        for (const cell of payload.accepted_cells ?? [])
            this._cell('heiwa-accepted', Number(cell), gw, gh, width, height);

        const bbox = payload.proposal_bbox;
        if (Array.isArray(bbox) && bbox.length === 4) {
            const [x1, y1, x2, y2] = bbox.map(Number);
            this._box('heiwa-proposal', x1 * width, y1 * height,
                Math.max((x2 - x1) * width, 4), Math.max((y2 - y1) * height, 4));
        }
        const xy = payload.proposal_xy;
        if (Array.isArray(xy) && xy.length === 2)
            this._box('heiwa-proposal', Number(xy[0]) * width - 9, Number(xy[1]) * height - 9, 18, 18);

        const reading = payload.reading_bbox;
        if (Array.isArray(reading) && reading.length === 4) {
            const [x1, y1, x2, y2] = reading.map(Number);
            this._box('heiwa-fovea', x1 * width, y1 * height,
                Math.max((x2 - x1) * width, 4), Math.max((y2 - y1) * height, 4));
        }
        const focus = payload.visual_focus_xy;
        if (Array.isArray(focus) && focus.length === 2)
            this._box('heiwa-focus-point', Number(focus[0]) * width - 6,
                Number(focus[1]) * height - 6, 12, 12);

        const guided = payload.guided_xy;
        if (Array.isArray(guided) && guided.length === 2) {
            const radius = Number(payload.search_radius ?? 0) * Math.min(width, height);
            this._box('heiwa-guided', Number(guided[0]) * width - radius,
                Number(guided[1]) * height - radius, 2 * radius, 2 * radius);
        }
        for (const point of payload.guidance_trace ?? []) {
            if (Array.isArray(point) && point.length === 2)
                this._box('heiwa-trace', Number(point[0]) * width - 4, Number(point[1]) * height - 4, 8, 8);
        }

        const held = Boolean(payload.f23_held);
        const readingText = String(payload.reading_text ?? '').replace(/\s+/g, ' ').slice(0, 80);
        const readingConfidence = Number(payload.reading_confidence ?? 0).toFixed(2);
        const label = new St.Label({
            text: `READ[${readingConfidence}]: ${readingText || '∅'}   F21=OK F22=wrong   F23 cursor erase: ${held ? 'HELD' : 'released'}   F24=cursor accept`,
            style_class: held ? 'heiwa-status-held' : 'heiwa-status',
            reactive: false,
        });
        label.set_position(14, 14);
        this._root.add_child(label);
    }

    _publishPointer() {
        const monitor = Main.layoutManager.primaryMonitor;
        if (!monitor)
            return;
        const [x, y] = global.get_pointer();
        const nx = Math.max(0, Math.min(1, (x - monitor.x) / Math.max(monitor.width - 1, 1)));
        const ny = Math.max(0, Math.min(1, (y - monitor.y) / Math.max(monitor.height - 1, 1)));
        const value = JSON.stringify({
            format: 1,
            updated_monotonic_us: GLib.get_monotonic_time(),
            screen: [monitor.width, monitor.height],
            xy: [nx, ny],
        });
        try {
            Gio.File.new_for_path(POINTER_PATH).replace_contents(
                new TextEncoder().encode(value), null, false,
                Gio.FileCreateFlags.REPLACE_DESTINATION, null);
        } catch (_error) {
            // Diagnostics must never disrupt the Shell.
        }
    }
}
