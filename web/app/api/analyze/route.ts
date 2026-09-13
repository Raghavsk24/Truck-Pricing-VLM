import { NextResponse } from "next/server";
import { resizeToJpeg } from "@/lib/image";
import { persistAndRespond, uploadJpeg } from "@/lib/pipeline";
import { extractFeaturesFromJpeg } from "@/lib/vlm";
import type { AnalyzeResponse } from "@/lib/types";

export const runtime = "nodejs";
export const maxDuration = 60;

function fail(message: string, status = 400) {
  const body: AnalyzeResponse = { status: "error", user_message: message };
  return NextResponse.json(body, { status });
}

export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const file = form.get("image");
    if (!(file instanceof File)) {
      return fail("Please upload a JPEG, PNG, or WebP image of your truck.");
    }
    if (file.size > 8 * 1024 * 1024) {
      return fail("That file is larger than 8 MB. Please upload a smaller photo.");
    }
    const type = file.type || "";
    if (!/^image\/(jpeg|png|webp|jpg)$/i.test(type) && !/\.(jpe?g|png|webp)$/i.test(file.name)) {
      return fail("Please upload a JPEG, PNG, or WebP image of your truck.");
    }

    const bytes = Buffer.from(await file.arrayBuffer());
    let jpeg: Buffer;
    try {
      jpeg = await resizeToJpeg(bytes);
    } catch {
      return fail("Could not read that image. Please try a different photo.");
    }

    const { path } = await uploadJpeg(jpeg);
    const vlm = await extractFeaturesFromJpeg(jpeg);
    const result = await persistAndRespond(path, vlm);
    return NextResponse.json(result);
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Something went wrong while inspecting the photo.";
    console.error(err);
    return fail(message, 500);
  }
}
