"use client";

export function AppHeader({
  showNew,
  onLogo,
  onNew,
}: {
  showNew: boolean;
  onLogo: () => void;
  onNew: () => void;
}) {
  return (
    <header className="header no-print">
      <div className="container header-inner">
        <button type="button" className="brand" onClick={onLogo}>
          Truck Pricing VLM
        </button>
        {showNew && (
          <button type="button" className="btn btn-secondary btn-sm" onClick={onNew}>
            New appraisal
          </button>
        )}
      </div>
    </header>
  );
}
