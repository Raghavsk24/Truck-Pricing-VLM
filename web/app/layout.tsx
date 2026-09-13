import type { Metadata } from "next";
import { Barlow, Barlow_Condensed } from "next/font/google";
import { AppToaster } from "@/components/AppToaster";
import "./globals.css";

const barlow = Barlow({
  variable: "--font-barlow",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});

const barlowCondensed = Barlow_Condensed({
  variable: "--font-barlow-condensed",
  subsets: ["latin"],
  weight: ["400", "600"],
});

export const metadata: Metadata = {
  title: "Blueoop — Truck appraisal",
  description:
    "Free Class 7 and 8 truck appraisal from photographs. A range, a condition breakdown, and an asking price you can defend.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${barlow.variable} ${barlowCondensed.variable} h-full antialiased`}
    >
      <body className="h-full">
        {children}
        <AppToaster />
      </body>
    </html>
  );
}
