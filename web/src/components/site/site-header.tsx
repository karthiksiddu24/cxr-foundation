"use client";

import { ShieldAlertIcon } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

import { LogoMark } from "./logo";
import { ThemeToggle } from "./theme-toggle";

const NAV = [
  { href: "/", label: "Analyze", short: "Analyze" },
  { href: "/model", label: "Model card", short: "Model" },
  { href: "/about", label: "Methodology", short: "Method" },
];

export const DISCLAIMER =
  "Research demonstration only. Not a medical device. Not for clinical use.";

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b bg-background/90 backdrop-blur supports-[backdrop-filter]:bg-background/75">
      <div className="mx-auto flex h-14 max-w-[1600px] items-center gap-2 px-3 sm:gap-4 sm:px-6">
        <Link
          href="/"
          aria-label="CXR-Foundation home"
          className="flex shrink-0 items-center gap-2 rounded-sm font-semibold tracking-tight"
        >
          <LogoMark className="size-6" />
          <span className="hidden sm:inline">CXR-Foundation</span>
        </Link>
        <nav
          aria-label="Main"
          className="flex min-w-0 items-center gap-0.5 text-sm sm:ml-2 sm:gap-1"
        >
          {NAV.map((n) => {
            const active =
              n.href === "/"
                ? pathname === "/" || pathname === "/viewer"
                : pathname.startsWith(n.href);
            return (
              <Link
                key={n.href}
                href={n.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "whitespace-nowrap rounded-md px-2 py-1.5 text-muted-foreground transition-colors hover:text-foreground sm:px-2.5",
                  active && "bg-secondary text-foreground",
                )}
              >
                <span className="sm:hidden">{n.short}</span>
                <span className="hidden sm:inline">{n.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-1">
          <ThemeToggle />
        </div>
      </div>
      <div role="note" className="border-t border-flag/30 bg-flag-soft">
        <p className="mx-auto flex max-w-[1600px] items-center gap-2 px-4 py-1.5 text-xs font-medium text-foreground sm:px-6">
          <ShieldAlertIcon className="size-3.5 shrink-0 text-flag" aria-hidden />
          {DISCLAIMER}
        </p>
      </div>
    </header>
  );
}
