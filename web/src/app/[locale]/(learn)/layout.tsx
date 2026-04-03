import { Sidebar } from "@/components/layout/sidebar";

const GOTTY_URL = process.env.NEXT_PUBLIC_GOTTY_URL || "http://127.0.0.1:8080/";

export default function LearnLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex gap-8">
      <Sidebar />
      <div className="min-w-0 flex-1">{children}</div>
      <aside className="hidden w-[420px] shrink-0 xl:block">
        <div className="sticky top-24 overflow-hidden rounded-xl border border-[var(--color-border)] bg-[var(--color-panel)]">
          <div className="border-b border-[var(--color-border)] px-4 py-2 text-sm font-medium text-[var(--color-text-muted)]">
            Terminal
          </div>
          <iframe
            src={GOTTY_URL}
            title="Gotty Terminal"
            className="h-[calc(100vh-8rem)] w-full border-0"
          />
        </div>
      </aside>
    </div>
  );
}
