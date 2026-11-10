export function LogoMark({ className }: { className?: string }) {
  // Abstract rib-cage mark: two lung fields bracketing a spine line.
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={className}>
      <rect x="1" y="1" width="22" height="22" rx="5" className="fill-primary/15" />
      <path d="M12 5v14" className="stroke-primary" strokeWidth="1.6" strokeLinecap="round" />
      <path
        d="M10 7.5C7 7.5 5.5 10.5 5.5 14c0 2 1 3.5 2.6 3.5 1.2 0 1.9-.9 1.9-2.4V7.5ZM14 7.5c3 0 4.5 3 4.5 6.5 0 2-1 3.5-2.6 3.5-1.2 0-1.9-.9-1.9-2.4V7.5Z"
        className="stroke-primary"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}
