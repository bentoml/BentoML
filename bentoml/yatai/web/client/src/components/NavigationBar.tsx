import * as React from "react";
import { useLocation, Link } from "react-router-dom";
import {
  Navbar,
  NavbarGroup,
  NavbarHeading,
  NavbarDivider,
  Alignment,
  AnchorButton,
  Classes,
} from "@blueprintjs/core";

import logo from "../assets/bentoml-logo.png";

const NavigationBar = () => {
  const [highlight, setHighlight] = React.useState("");
  const location = useLocation();

  React.useEffect(() => {
    const currentLocation = location.pathname.split("/")[1];
    setHighlight(currentLocation);
  }, [location]);

  return (
    <Navbar style={{ paddingLeft: "10%", marginLeft: 0 }}>
      <NavbarGroup align={Alignment.LEFT}>
        <NavbarHeading>
          <Link to="/">
            <img src={`/${logo}`} width={150}  alt={""}/>{" "}
          </Link>
        </NavbarHeading>
        <NavbarDivider />
        <AnchorButton
          className={Classes.MINIMAL}
          large
          text="Repository"
          href="/repository"
          style={getHighlightStyle(highlight, "repository")}
        />
        <AnchorButton
          className={Classes.MINIMAL}
          large
          text="Deployments"
          href="/deployments"
          style={getHighlightStyle(highlight, "deployments")}
        />
      </NavbarGroup>
    </Navbar>
  );
};

const getHighlightStyle = (highlight, path) => ({
  fontWeight: highlight === path ? ("bold" as "bold") : ("normal" as "normal"),
});

export default NavigationBar;
