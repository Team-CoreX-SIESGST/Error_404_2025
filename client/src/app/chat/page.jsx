'use client'

import ArenaChatSection from "@/components/sections/ArenaChatSection"
import Navbar from "@/components/Navbar"
import WarpStreamEffect from "@/components/WarpStreamEffect"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"

export default function ChatPage() {
  return (
    <div className="min-h-screen bg-background text-foreground overflow-x-hidden relative">
      {/* Background Effect */}
      <WarpStreamEffect className="z-0" />
      
      {/* Content */}
      <div className="relative z-10">
        <Navbar />
        <div className="pt-20">
          {/* Back to Home Link */}
          <div className="container mx-auto px-4 md:px-8 lg:px-16 mb-6">
            <Link 
              href="/"
              className="inline-flex items-center gap-2 text-gray-400 hover:text-primary transition-colors group"
            >
              <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
              <span className="text-sm font-medium">Back to Home</span>
            </Link>
          </div>

          {/* Chat Section */}
          <ArenaChatSection showBackground={false} />
        </div>
      </div>
    </div>
  )
}

