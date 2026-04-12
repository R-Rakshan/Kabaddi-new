import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Kabaddi Vision — Player Tracking System",
  description:
    "AI-powered Kabaddi match analysis: YOLOv8 detection, ByteTrack tracking, team classification and pose-based player IDs.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
