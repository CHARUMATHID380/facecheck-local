'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { LockKeyhole, ShieldCheck, UserPlus, ScanFace } from 'lucide-react'

export default function Topbar() {
  const path = usePathname()

  return (
    <header className="topbar">
      <Link className="brand" href="/" aria-label="Verifai home">
        <span className="brand-mark"><ShieldCheck size={18} /></span>
        <span>verifai</span>
      </Link>

      <nav className="nav-links" aria-label="Main navigation">
        <Link
          href="/enroll"
          className={`nav-link${path === '/enroll' ? ' active' : ''}`}
        >
          <UserPlus size={14} /> Enroll
        </Link>
        <Link
          href="/identify"
          className={`nav-link${path === '/identify' ? ' active' : ''}`}
        >
          <ScanFace size={14} /> Identify
        </Link>
      </nav>

      <div className="privacy-pill"><LockKeyhole size={14} /> Local-only processing</div>
    </header>
  )
}
