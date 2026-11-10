/** Client-side pre-checks so obvious mistakes get instant, specific feedback.
 * The server re-validates everything (it never trusts the client). */

export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

export type FileKind = "png" | "jpeg" | "dicom";

export async function sniffFile(file: File): Promise<FileKind | null> {
  const head = new Uint8Array(await file.slice(0, 132).arrayBuffer());
  const png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (png.every((b, i) => head[i] === b)) return "png";
  if (head[0] === 0xff && head[1] === 0xd8 && head[2] === 0xff) return "jpeg";
  if (head.length >= 132 && String.fromCharCode(...head.slice(128, 132)) === "DICM") return "dicom";
  return null;
}

export type PrecheckResult = { ok: true; kind: FileKind } | { ok: false; message: string };

export async function precheck(file: File): Promise<PrecheckResult> {
  if (file.size === 0) return { ok: false, message: "That file is empty." };
  if (file.size > MAX_UPLOAD_BYTES)
    return {
      ok: false,
      message: `That file is ${(file.size / 1e6).toFixed(1)} MB; the limit is 25 MB.`,
    };
  const kind = await sniffFile(file);
  if (!kind) return { ok: false, message: "Unsupported file. Upload a PNG, JPEG, or DICOM image." };
  return { ok: true, kind };
}
