#import "@preview/charged-ieee:0.1.3": ieee

#show: ieee.with(
  title: [],
  abstract: [
    This is a cool abstract.
  ],
  authors: (
    (
      name: "Gavin Hua",
      organization: [California Institute of Technology],
      location: [Pasadena, CA],
      email: "ghua@caltech.edu"
    ),
    (
      name: "Gio Huh",
      organization: [California Institute of Technology],
      location: [Pasadena, CA],
      email: "ghuh@caltech.edu"
    ),
    (
      name: "Taekyung Lee",
      organization: [California Institute of Technology],
      location: [Pasadena, CA],
      email: "tklee2@caltech.edu"
    ),
  ),
  index-terms: ("SLAM", "Robotics"),
)