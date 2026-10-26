import type { EncodingInfo } from "@/lib/types";

type EncodingPickerProps = {
  encodings: EncodingInfo[];
  selected: string;
  onSelect: (name: string) => void;
};

/** A row of buttons to choose the tokenizer, with its description. */
export function EncodingPicker({
  encodings,
  selected,
  onSelect,
}: EncodingPickerProps) {
  const description = encodings.find((e) => e.name === selected)?.description;

  return (
    <div>
      <div className="flex flex-wrap gap-2">
        {encodings.map((encoding) => {
          const isSelected = encoding.name === selected;
          return (
            <button
              key={encoding.name}
              type="button"
              onClick={() => onSelect(encoding.name)}
              title={encoding.description}
              className={
                "rounded-full px-3.5 py-1.5 font-mono text-sm transition " +
                (isSelected
                  ? "bg-gradient-to-r from-violet-500 to-cyan-500 text-white shadow-lg shadow-violet-500/30"
                  : "border border-white/10 bg-white/5 text-slate-300 hover:border-white/30 hover:text-white")
              }
            >
              {encoding.name}
            </button>
          );
        })}
      </div>
      {description && (
        <p className="mt-2 text-sm text-slate-400">{description}</p>
      )}
    </div>
  );
}
