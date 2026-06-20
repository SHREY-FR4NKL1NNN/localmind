import { useEffect, useState } from 'react'

// Subscribe to a CSS media query and re-render when it changes. Used to switch
// the GatingDiagram between its horizontal (desktop) and vertical (mobile)
// layouts — a layout decision SVG can't make with CSS alone.
export function useMediaQuery(query) {
  const [matches, setMatches] = useState(
    () => typeof window !== 'undefined' && window.matchMedia(query).matches
  )

  useEffect(() => {
    const mql = window.matchMedia(query)
    const onChange = () => setMatches(mql.matches)
    onChange()
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [query])

  return matches
}
