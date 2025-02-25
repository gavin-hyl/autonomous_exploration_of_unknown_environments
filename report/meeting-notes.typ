#let title = "ME 133b Final Project Meeting Notes"
#let author = "Gio, TK, Gavin"
#let date = "2025-02"

#set page(
  numbering: "1",
    header: [
      #smallcaps([#title])
      #h(1fr) #author
      #line(length: 100%)
    ],
)

#set par(justify: true)

= Project Idea Discussion

== Multi-Robot Coordination

=== Overview
Our project proposes a multi-robot system designed for emergency rescue in a building on fire. The team’s approach leverages both aerial and ground platforms: multiple drones for rapid aerial reconnaissance and mapping, and a ground robot (akin to Boston Dynamics’ Spot) for on-the-ground rescue operations. The objective is to cover as much ground as possible, efficiently locate exit signs and survivors, and then coordinate to safely extract people from danger.

=== Challenges

- Mapping in a Hazardous Environment: Dealing with rapidly changing conditions due to fire, which may cause parts of the map to become outdated quickly.

- Inter-Robot Communication: Exploring methods for limited communication between the drones and the ground robot.

- Dynamic Re-Planning: Adapting to new obstacles and recalculating safe routes promptly to ensure the safety of both the rescue team and the survivors.