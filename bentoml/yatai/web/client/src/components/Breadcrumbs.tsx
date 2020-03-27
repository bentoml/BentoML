import * as React from "react";
import { useLocation } from "react-router-dom";
import { capitalize } from "lodash";

import {
  Breadcrumbs as BlueprintBreadcrumbs,
  IBreadcrumbProps
} from "@blueprintjs/core";
import { Section } from "../ui/Layout";

const HOME_CRUMB: IBreadcrumbProps = {
  text: "Home",
  href: "/"
};

const Breadcrumbs: React.FC = () => {
  const [breadcrumbs, setBreadcrumbs] = React.useState<Array<IBreadcrumbProps>>(
    []
  );
  const location = useLocation();

  React.useEffect(() => {
    const parsedBreadcrumbs = getBreadcrumbs(location.pathname);
    setBreadcrumbs(parsedBreadcrumbs);
  }, [location]);

  breadcrumbs.length > 0 && breadcrumbs.unshift(HOME_CRUMB);
  if (breadcrumbs.length === 0) {
    return null;
  }
  return (
    <Section>
      <BlueprintBreadcrumbs items={breadcrumbs} />
    </Section>
  );
};

const getBreadcrumbs = (pathname: string): Array<IBreadcrumbProps> => {
  const pathSnippets = pathname.split("/").filter(i => i);

  return pathSnippets.map((name, index) => {
    const isLastOne = index === pathSnippets.length - 1;
    const url = `/${pathSnippets.slice(0, index + 1).join("/")}`;
    const formattedName = capitalize(name);
    return isLastOne
      ? { text: formattedName }
      : { text: formattedName, href: url };
  });
};

export default Breadcrumbs;
