import { Activity, BookOpen, Boxes, LayoutDashboard, Network, type LucideIcon } from 'lucide-react'

export type SectionId = 'dashboard' | 'studio' | 'training' | 'explorer' | 'learning'

export interface Section {
  id: SectionId
  label: string
  icon: LucideIcon
  /** Plan milestone that delivers the section; null when available now. */
  milestone: string | null
  summary: string
}

export const SECTIONS: Section[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: LayoutDashboard,
    milestone: null,
    summary: 'System status and project overview.',
  },
  {
    id: 'studio',
    label: 'Architecture Studio',
    icon: Network,
    milestone: 'Milestones 2–3',
    summary: 'Build dense networks and explore them in synchronized 2D and 3D views.',
  },
  {
    id: 'training',
    label: 'Training Lab',
    icon: Activity,
    milestone: 'Milestone 4',
    summary: 'Train PyTorch networks and inspect real loss, activations, and gradients.',
  },
  {
    id: 'explorer',
    label: 'Model Explorer',
    icon: Boxes,
    milestone: 'Milestone 5',
    summary: 'Import ONNX models and inspect operators, shapes, and tensor summaries.',
  },
  {
    id: 'learning',
    label: 'Learning Mode',
    icon: BookOpen,
    milestone: 'Milestone 7',
    summary: 'Guided explanations of dense networks, CNNs, and transformers.',
  },
]
