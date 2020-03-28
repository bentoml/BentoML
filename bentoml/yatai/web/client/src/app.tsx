import * as React from "react";
import { Switch, Route } from "react-router";
import { BrowserRouter, Link } from "react-router-dom";
import {
  Navbar,
  NavbarGroup,
  NavbarHeading,
  NavbarDivider,
  Alignment,
  Button,
  Classes
} from "@blueprintjs/core";
import HomePage from "./pages/HomePage";
import { DeploymentsList } from "./pages/DeploymentsList";
import { DeploymentDetails } from "./pages/DeploymentDetails";
import Repository from "./pages/Repository";
import { BentoServicesList } from "./pages/BentoServiceList";
import { BentoServiceDetail } from "./pages/BentoServiceDetail";
import Layout from "./ui/Layout";
import Breadcrumbs from "./components/Breadcrumbs";
import logo from "./assets/bentoml-logo.png";

const NavigationBar = () => (
  <Navbar style={{ paddingLeft: "10%", marginLeft: 0 }}>
    <NavbarGroup align={Alignment.LEFT}>
      <Link to="/">
        <NavbarHeading>
          <img src={logo} width={150} />
        </NavbarHeading>
      </Link>
      <NavbarDivider />
      <Link to="/repository">
        <Button className={Classes.MINIMAL} text="Repository" />
      </Link>
      <Link to="/deployments">
        <Button className={Classes.MINIMAL} text="Deployments" />
      </Link>
    </NavbarGroup>
  </Navbar>
);

export const App = () => (
  <BrowserRouter>
    <NavigationBar />
    <Layout>
      <Breadcrumbs />
      <div>
        <Switch>
          <Route
            path="/deployments/:namespace/:name"
            component={DeploymentDetails}
          />
          <Route path="/deployments/:namespace" component={DeploymentsList} />
          <Route path="/deployments" component={DeploymentsList} />

          <Route
            path="/repository/:name/:version"
            component={BentoServiceDetail}
          />
          <Route path="/repository/:name" component={BentoServicesList} />
          <Route path="/repository" component={Repository} />

          <Route path="/about" component={HomePage} />
          <Route path="/config" component={HomePage} />
          <Route exact path="/" component={HomePage} />
        </Switch>
      </div>
    </Layout>
  </BrowserRouter>
);
