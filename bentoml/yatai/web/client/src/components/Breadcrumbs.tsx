import * as React from "react";
import { useLocation } from "react-router-dom";
import {
  Breadcrumbs as BlueprintBreadcrumbs,
  IBreadcrumbProps
} from "@blueprintjs/core";

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
  return <BlueprintBreadcrumbs items={breadcrumbs} />;
};

const getBreadcrumbs = (pathname: string): Array<IBreadcrumbProps> => {
  const pathSnippets = pathname.split("/").filter(i => i);

  return pathSnippets.map((name, index) => {
    const isLastOne = index === pathSnippets.length - 1;
    const url = `/${pathSnippets.slice(0, index + 1).join("/")}`;
    return isLastOne ? { text: name } : { text: name, href: url };
  });
};

export default Breadcrumbs;
