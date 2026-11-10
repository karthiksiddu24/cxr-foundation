"use client";

import * as React from "react";

export type ShortcutMap = Record<string, (e: KeyboardEvent) => void>;

/** Global keyboard shortcuts, ignored while typing in form fields or when a modifier is held. */
export function useShortcuts(map: ShortcutMap, enabled = true): void {
  const ref = React.useRef(map);
  React.useEffect(() => {
    ref.current = map;
  });
  React.useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName))) return;
      if (t?.closest("[role=dialog]")) return;
      const handler = ref.current[e.key];
      if (handler) {
        e.preventDefault();
        handler(e);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled]);
}
