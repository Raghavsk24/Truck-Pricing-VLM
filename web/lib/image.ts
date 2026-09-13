import "server-only";
import sharp from "sharp";

export const LONG_EDGE = 1568;
export const JPEG_QUALITY = 85;

export async function resizeToJpeg(input: Buffer): Promise<Buffer> {
  return sharp(input)
    .rotate()
    .resize({
      width: LONG_EDGE,
      height: LONG_EDGE,
      fit: "inside",
      withoutEnlargement: true,
    })
    .jpeg({ quality: JPEG_QUALITY, mozjpeg: true })
    .toBuffer();
}
