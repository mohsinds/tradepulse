import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TradePulse",
  description: "Trading research and execution dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 min-h-screen">
        {children}
      </body>
    </html>
  );
}
