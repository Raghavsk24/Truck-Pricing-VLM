import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { AppToaster } from "@/components/AppToaster";
import "./globals.css";

const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Truck Pricing VLM — Truck appraisal",
  description:
    "Free Class 7 and 8 truck appraisal from photographs. A range, a condition breakdown, and an asking price you can defend.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${inter.variable} h-full antialiased`}
    >
      <body className="h-full">
        {children}
        <AppToaster />
      </body>
    </html>
  );
}
