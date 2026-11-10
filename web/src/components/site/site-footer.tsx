export function SiteFooter() {
  return (
    <footer className="border-t">
      <div className="mx-auto flex max-w-[1600px] flex-col gap-1 px-4 py-4 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <p>
          CXR-Foundation · research demonstration · images are processed in memory and not stored.
        </p>
        <p>Sample images: NIH ChestX-ray14 (NIH Clinical Center).</p>
      </div>
    </footer>
  );
}
