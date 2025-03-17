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

= Project Ideas

== Multi-Robot Coordination

=== Overview
Our project proposes a multi-robot system designed for emergency rescue in a building on fire. The team's approach leverages both aerial and ground platforms: multiple drones for rapid aerial reconnaissance and mapping, and a ground robot (akin to Boston Dynamics' Spot) for on-the-ground rescue operations. The objective is to cover as much ground as possible, efficiently locate exit signs and survivors, and then coordinate to safely extract people from danger.

=== Challenges

- Mapping in a Hazardous Environment: Dealing with rapidly changing conditions due to fire, which may cause parts of the map to become outdated quickly.

- Inter-Robot Communication: Exploring methods for limited communication between the drones and the ground robot.

- Dynamic Re-Planning: Adapting to new obstacles and recalculating safe routes promptly to ensure the safety of both the rescue team and the survivors.


= Project Idea Discussion

== Multi-Robot Coordination
- Gunter doesn't like the two parts, nor does he like SLAM.
- If we do SLAM, we should focus on it.
- What features are we going to detect?
- Imperfect odometry for the robots (considered to be UAVs?).
- GraphSLAM, need decent data.
- by friday :D
- multi-robot is ok, SLAM alone is enuf (robots exchange full information)

==
Combine LIDAR, IMU, Beacon, and physics simulation nodes. (One simulation, one robot node)
We can use RVIZ
Set up occupancy grid and learn to populate it with the LIDAR.